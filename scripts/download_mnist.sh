#!/usr/bin/env sh
# Downloads the MNIST dataset into data/ (about 55 MB unpacked).
set -e
cd "$(dirname "$0")/../data"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "$f" ]; then
    echo "downloading $f"
    curl -sSfL "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz" | gunzip > "$f"
  fi
done
echo "MNIST is ready in data/"
