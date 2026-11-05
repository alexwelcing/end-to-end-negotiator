#!/usr/bin/env bash
# Bundle the Worker and serve it on workerd at http://127.0.0.1:8788
set -euo pipefail
cd "$(dirname "$0")/.."
npx esbuild src/index.ts --bundle --format=esm --outfile=dist/index.js --log-level=warning
mkdir -p .do-storage export
exec npx workerd serve workerd.capnp "$@"
