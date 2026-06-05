#!/usr/bin/env bash
set -euo pipefail

# Runs the figure reproduction scripts in order.
# OUT_DIR is treated as the root output directory for this wrapper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${OUT_DIR:-${BASE_DIR}/scripts/result_figure}}"

run_figure() {
  local label="$1"
  local script_name="$2"
  local output_subdir="$3"

  echo "=== ${label}: ${script_name} ==="
  OUT_DIR="${OUT_ROOT}/${output_subdir}" bash "${SCRIPT_DIR}/${script_name}"
  echo "=== ${label} completed: ${OUT_ROOT}/${output_subdir} ==="
}

run_figure "Figure 9 workload matrix" "figure9_real_workloads.sh" "figure9_real_workloads"
run_figure "Figure 10 read breakdown" "figure10_read_breakdown.sh" "figure10_read_breakdown"
run_figure "Figure 11 write metrics" "figure11_write_metrics.sh" "figure11_write_metrics"
run_figure "Figure 12 model training" "figure12_model_training.sh" "figure12_model_training"

echo "All figure reproduction scripts completed. Outputs are under ${OUT_ROOT}"
