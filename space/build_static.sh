#!/bin/sh
# Build the free static Space (in-browser Pyodide) into $1: web/ + demo_core.py + the repo's genesis/ package, unchanged.
set -e; cd "$(dirname "$0")"; D=${1:-build}; rm -rf "$D"; mkdir -p "$D/genesis"
cp web/index.html web/worker.js demo_core.py "$D"/; cp ../genesis/*.py "$D/genesis/"
cat > "$D/README.md" <<'MD'
---
title: GENESIS demo
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Tiny evolving town of learning minds, runs in your browser
---
Interactive demo of [GENESIS](https://github.com/guajardoblake/genesis) (Apache-2.0). Runs the repo's engine unchanged
in your browser via Pyodide. Toy world; not evidence about AGI.
MD
