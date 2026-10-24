#!/bin/sh
# Free deploy: static Space guajardo/genesis-demo (Pyodide in-browser). Needs HF auth (hf auth login or HF_TOKEN).
set -e; cd "$(dirname "$0")"; T=$(mktemp -d); ./build_static.sh "$T"
hf repos create guajardo/genesis-demo --repo-type space --space-sdk static --exist-ok
hf upload guajardo/genesis-demo "$T" . --repo-type space --commit-message "Deploy GENESIS in-browser demo"
