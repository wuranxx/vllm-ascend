#!/usr/bin/env bash
# Compatibility alias. New automation should call run_remote_validation.sh.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_remote_validation.sh" "$@"
