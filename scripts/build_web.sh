#!/usr/bin/env sh
# Builds the WebAssembly module and copies it, plus model.bin, into docs/ for GitHub Pages.
set -e
cd "$(dirname "$0")/.."
rustup target add wasm32-unknown-unknown
cargo build --lib --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/rustvision.wasm docs/
if [ -f model.bin ]; then cp model.bin docs/model.bin; fi
echo "docs/ is ready. Preview with: python3 -m http.server -d docs 8000"
