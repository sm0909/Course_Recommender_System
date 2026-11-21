#!/usr/bin/env bash
# One command to reproduce the numbers in results/ (works with or without a virtualenv).
set -euo pipefail
cd "$(dirname "$0")"
PIP_FLAGS="--user"; [ -n "${VIRTUAL_ENV:-}" ] && PIP_FLAGS=""   # --user is not allowed inside a venv
python3 -m pip install $PIP_FLAGS -q -r requirements-eval.txt
python3 evaluate_recommenders.py "$@"
