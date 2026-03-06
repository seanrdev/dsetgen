"""SQLite-backed checkpoint manager.

If the pipeline crashes at file 45,000 of 50,000, this module ensures
we can restart and skip directly to file 45,001.

Schema
------
The SQLite database stores one row per *file* that has been **fully
processed** (all chunks sent to the LLM and the output records written).
A file is only marked complete once the full cycle succeeds — partial
work is discarded on restart.

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS checkpoints (
       file_path  TEXT PRIMARY KEY,
       status     TEXT NOT NULL DEFAULT 'complete',
       num_chunks INTEGER,
       num_records INTEGER,
       finished_at TEXT,
       error_msg  TEXT
   );

   CREATE TABLE IF NOT EXISTS run_metadata (
       key   TEXT PRIMARY KEY,
       value TEXT
   );

Usage
-----
>>> ckpt = CheckpointManager("state.db")
>>> ckpt.initialize()
>>> if not ckpt.is_complete("/data/raw/report.pdf"):
...     process(report)
...     ckpt.mark_complete("/data/raw/report.pdf", num_chunks=12, num_records=12)
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional, Set

from doc2dataset.exceptions import CheckpointError

logger = logging.getLogger(__name__)


class CheckpointManager:
    """ACID-safe checkpoint store backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Created on first use.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Open (or create) the database and ensure the schema exists."""
        try:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")  # safe for crash
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    file_path    TEXT PRIMARY KEY,
                    status       TEXT NOT NULL DEFAULT 'complete',
                    num_chunks   INTEGER,
                    num_records  INTEGER,
                    finished_at  TEXT,
                    error_msg    TEXT
                );

                CREATE TABLE IF NOT EXISTS run_metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            self._conn.commit()
            logger.info("Checkpoint DB ready at %s", self._db_path)
        except sqlite3.Error as exc:
            raise CheckpointError(f"Cannot initialise checkpoint DB: {exc}") from exc

    def close(self) -> None:
        """Commit outstanding transactions and close the connection."""
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    # ── Query ──────────────────────────────────────────────────────────────

    def is_complete(self, file_path: str) -> bool:
        """Return ``True`` if *file_path* was fully processed previously."""
        row = self._ensure_conn().execute(
            "SELECT 1 FROM checkpoints WHERE file_path = ? AND status = 'complete'",
            (file_path,),
        ).fetchone()
        return row is not None

    def is_failed(self, file_path: str) -> bool:
        """Return ``True`` if *file_path* previously failed."""
        row = self._ensure_conn().execute(
            "SELECT 1 FROM checkpoints WHERE file_path = ? AND status = 'failed'",
            (file_path,),
        ).fetchone()
        return row is not None

    def completed_files(self) -> Set[str]:
        """Return the set of all file paths marked complete."""
        rows = self._ensure_conn().execute(
            "SELECT file_path FROM checkpoints WHERE status = 'complete'"
        ).fetchall()
        return {r[0] for r in rows}

    def stats(self) -> dict:
        """Return aggregate counts for logging."""
        conn = self._ensure_conn()
        total = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        ok = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE status='complete'"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE status='failed'"
        ).fetchone()[0]
        return {"total": total, "complete": ok, "failed": failed}

    # ── Mutation ───────────────────────────────────────────────────────────

    def mark_complete(
        self,
        file_path: str,
        *,
        num_chunks: int = 0,
        num_records: int = 0,
    ) -> None:
        """Record that *file_path* was fully processed."""
        self._ensure_conn().execute(
            """
            INSERT INTO checkpoints (file_path, status, num_chunks, num_records, finished_at)
            VALUES (?, 'complete', ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                status      = 'complete',
                num_chunks  = excluded.num_chunks,
                num_records = excluded.num_records,
                finished_at = excluded.finished_at
            """,
            (
                file_path,
                num_chunks,
                num_records,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()  # type: ignore[union-attr]

    def mark_failed(self, file_path: str, error_msg: str = "") -> None:
        """Record that *file_path* failed (so we can optionally retry later)."""
        self._ensure_conn().execute(
            """
            INSERT INTO checkpoints (file_path, status, error_msg, finished_at)
            VALUES (?, 'failed', ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                status      = 'failed',
                error_msg   = excluded.error_msg,
                finished_at = excluded.finished_at
            """,
            (file_path, error_msg, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()  # type: ignore[union-attr]

    def mark_batch_complete(
        self,
        entries: list[tuple[str, int, int]],
    ) -> None:
        """Atomically mark multiple files complete in one transaction.

        Parameters
        ----------
        entries:
            List of ``(file_path, num_chunks, num_records)`` tuples.
        """
        conn = self._ensure_conn()
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction():
            conn.executemany(
                """
                INSERT INTO checkpoints (file_path, status, num_chunks, num_records, finished_at)
                VALUES (?, 'complete', ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    status      = 'complete',
                    num_chunks  = excluded.num_chunks,
                    num_records = excluded.num_records,
                    finished_at = excluded.finished_at
                """,
                [(fp, nc, nr, now) for fp, nc, nr in entries],
            )

    def set_metadata(self, key: str, value: str) -> None:
        """Store an arbitrary key-value pair (e.g. run ID, config hash)."""
        self._ensure_conn().execute(
            "INSERT OR REPLACE INTO run_metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()  # type: ignore[union-attr]

    def get_metadata(self, key: str) -> Optional[str]:
        row = self._ensure_conn().execute(
            "SELECT value FROM run_metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def reset(self) -> None:
        """Drop all checkpoint data (useful for fresh reruns)."""
        conn = self._ensure_conn()
        conn.execute("DELETE FROM checkpoints")
        conn.execute("DELETE FROM run_metadata")
        conn.commit()
        logger.warning("Checkpoint DB reset — all progress cleared")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise CheckpointError("CheckpointManager not initialised — call .initialize() first")
        return self._conn

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        conn = self._ensure_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
