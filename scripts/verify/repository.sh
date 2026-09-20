#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python -m unittest discover -s tests/architecture -v
python architecture/check_repository_state.py
python architecture/check_claims.py
python architecture/check_methods.py
python architecture/check_modules.py
python architecture/check_planning.py
python architecture/check_experiments.py
python architecture/check_semantic_defaults.py
python architecture/check_documentation_quality.py
python architecture/check_durable_text.py
python architecture/check_data_authorities.py
python architecture/check_station_providers.py
python architecture/check_configurations.py
python architecture/check_hazards.py
python architecture/check_external_evaluations.py
python architecture/check_sheaf_realization.py
python architecture/check_root_layout.py
python architecture/source_gates.py --strict
python architecture/render_state.py --check
python architecture/render_roadmap.py --check
