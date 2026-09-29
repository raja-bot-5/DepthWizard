#!/usr/bin/env bash
# Package the code (no data, no env, no weights) for upload as a Kaggle Dataset named "depthwizard-code".
#   bash scripts/make_kaggle_bundle.sh      -> dist/depthwizard-code.zip
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/depthwizard-code.zip
zip -qr dist/depthwizard-code.zip src training configs scripts tests setup/requirements.txt setup/requirements.lock.txt \
    docs/gamus_audit.md CLAUDE.md -x '*/__pycache__/*' '*.pyc'
echo "wrote dist/depthwizard-code.zip ($(du -h dist/depthwizard-code.zip | cut -f1))"
echo "Upload it at kaggle.com -> Datasets -> New Dataset, name: depthwizard-code"
