#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

ABLATION_NAME="${1:-}"
OUTPUT_PATH="${2:-}"
[[ -n "${ABLATION_NAME}" && -n "${OUTPUT_PATH}" ]] || {
  printf 'usage: bash scripts/materialize_ablation.sh NAME OUTPUT.yaml\n' >&2
  exit 2
}

MATRIX_PATH="${PROJECT_ROOT}/configs/ablations.yaml"
ABLATION_NAME="${ABLATION_NAME}" OUTPUT_PATH="${OUTPUT_PATH}" MATRIX_PATH="${MATRIX_PATH}" python3 - <<'PY'
import copy
import os
from pathlib import Path

import yaml

matrix_path = Path(os.environ["MATRIX_PATH"])
matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
name = os.environ["ABLATION_NAME"]
try:
    ablation = matrix["ablations"][name]
except KeyError as exc:
    available = ", ".join(sorted(matrix.get("ablations", {})))
    raise SystemExit(f"unknown ablation {name!r}; available: {available}") from exc

base_path = (matrix_path.parent / matrix["base_config"]).resolve()
config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
for dotted_key, value in ablation.get("overrides", {}).items():
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = copy.deepcopy(value)

config["experiment_name"] = f"{config['experiment_name']}-{name}"
config["ablation"] = {
    "name": name,
    "description": ablation.get("description"),
    "evaluation_only": bool(ablation.get("evaluation_only", False)),
}
for field in ("checkpoint",):
    if field in ablation:
        config["ablation"][field] = copy.deepcopy(ablation[field])
output = Path(os.environ["OUTPUT_PATH"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(output.resolve())
PY
