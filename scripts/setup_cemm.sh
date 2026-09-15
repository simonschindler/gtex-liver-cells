#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LIVERCELLS_VENV:-${TMPDIR:-/tmp}/gtex-liver-cells-venv}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p "$(dirname "${VENV_DIR}")"
uv venv --python 3.12 --allow-existing "${VENV_DIR}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
uv sync --project "${REPO_DIR}"

echo "Environment ready at ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
