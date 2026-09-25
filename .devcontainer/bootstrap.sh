#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${CODESPACES:-false}" == "true" ]]; then
  if [[ -z "${CLIMATE_CODESPACES_AUTHORIZATION:-}" ]]; then
    echo "Codespaces bootstrap refused: no finite-resource authorization is set." >&2
    exit 64
  fi
  python architecture/finite_resources.py claim-codespaces \
    --authorization-id "$CLIMATE_CODESPACES_AUTHORIZATION"
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  gfortran=4:13.2.0-7ubuntu1 \
  libblas-dev=3.12.0-3build1 \
  liblapack-dev=3.12.0-3build1 \
  pkg-config
sudo rm -rf /var/lib/apt/lists/*

python -m pip install --disable-pip-version-check \
  -r requirements/reference-ebm-representation.txt

go install cuelang.org/go/cmd/cue@v0.17.1
sudo ln -sf "$(go env GOPATH)/bin/cue" /usr/local/bin/cue

python architecture/check_experiments.py
python architecture/codespaces_campaign.py --list
