#!/bin/bash

# 1. Install the system dependencies required to compile psycopg2 (for pgcli)
sudo apt update
sudo apt install -y libpq-dev python3-dev build-essential

# 2. Install the top-level applications using pipx
pipx install "mitmproxy==12.2.1"
pipx install "bloodhound==1.9.0"
pipx install "pgcli==4.4.0"
pipx install "faraday-agent-dispatcher==3.2.1"
pipx install "faradaysec==5.19.0"

# 3. Install theHarvester directly from source to avoid PyPI versioning issues
pipx install git+https://github.com/laramies/theHarvester.git