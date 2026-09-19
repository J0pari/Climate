#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  gfortran \
  libblas-dev \
  liblapack-dev \
  pkg-config
sudo rm -rf /var/lib/apt/lists/*

python -m pip install --disable-pip-version-check \
  -r requirements/reference-ebm-representation.txt

go install cuelang.org/go/cmd/cue@v0.17.1
sudo ln -sf "$(go env GOPATH)/bin/cue" /usr/local/bin/cue

python architecture/check_experiments.py
python architecture/codespaces_campaign.py --list
