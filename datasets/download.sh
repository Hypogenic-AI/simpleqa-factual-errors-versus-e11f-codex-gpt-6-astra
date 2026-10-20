#!/usr/bin/env bash
set -euo pipefail
pwd
test -d .git
test -x .venv/bin/python
.venv/bin/python scripts/prepare_simpleqa.py
