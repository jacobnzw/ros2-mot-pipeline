#!/usr/bin/env bash
# Stop on command errors (-e), unset variables (-u), and failed commands inside
# a pipeline (-o pipefail), so an unsuccessful export/import can't look successful.
set -euo pipefail

# Print usage errors to stderr (>&2), keeping normal output separate.
usage() {
  echo "Usage: $0 path/to/recording.rrd [run-id]" >&2
}

# $# is the number of command-line arguments. Accept the recording path and an
# optional run label; [[ ... ]] is Bash's conditional syntax.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

# Check required programs before doing any work. command -v returns non-zero if
# a program is unavailable; >/dev/null hides its normal output.
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required to run scripts/rrd_to_csv.py" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Error: Docker with the Compose plugin is required" >&2
  exit 1
fi

# Resolve the input to an absolute path, so later commands work from any
# directory. Quoting "$1" preserves paths containing spaces; -- ends options
# in case a filename starts with a dash.
recording_path=$(realpath -- "$1")
if [[ ! -f "$recording_path" ]]; then
  echo "Error: recording file does not exist: $recording_path" >&2
  exit 1
fi
# ${recording_path,,} lowercases the path, allowing .RRD as well as .rrd.
if [[ ${recording_path,,} != *.rrd ]]; then
  echo "Error: expected a .rrd recording: $recording_path" >&2
  exit 1
fi

# Find the repository relative to this script, not relative to the caller's
# current directory. BASH_SOURCE[0] is this script's filename; dirname and
# basename split paths into their directory and final filename.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
recording_dir=$(dirname -- "$recording_path")
# Remove the .rrd suffix to get the filename stem. The run label can be
# provided explicitly; otherwise use the run directory for recording.rrd or
# the filename stem for other recording names.
recording_stem=$(basename -- "${recording_path%.rrd}")
if [[ $# -eq 2 ]]; then
  run_id=$2
elif [[ $recording_stem == recording ]]; then
  run_id=$(basename -- "$recording_dir")
else
  run_id=$recording_stem
fi
# A blank second argument would create unlabeled/misleading time series.
if [[ -z $run_id ]]; then
  echo "Error: run-id must not be empty" >&2
  exit 1
fi

csv_path="$recording_dir/latency.csv"
kpi_csv_path="$recording_dir/trackeval.csv"

# Run commands from the repository root so Compose can find its config and
# related provisioning files. uv uses the project environment for Rerun's
# Python dependencies; each argument is quoted to keep paths with spaces intact.
cd -- "$repo_root"
echo "==> Exporting latency samples from $recording_path (run: $run_id)"
uv run --project "$repo_root" python "$repo_root/scripts/rrd_to_csv.py" \
  "$recording_path" --output "$csv_path" --run-id "$run_id"

# TrackEval is an optional post-processing step. When its detailed result files
# are present in the run folder, export their per-sequence and combined KPIs.
trackeval_file=$(find "$recording_dir" -type f -name '*_detailed.csv' -print -quit)
if [[ -n $trackeval_file ]]; then
  echo "==> Exporting TrackEval KPIs from $recording_dir"
  uv run --project "$repo_root" python "$repo_root/scripts/trackeval_to_csv.py" \
    "$recording_dir" --output "$kpi_csv_path" --run-id "$run_id"
else
  echo "==> No TrackEval detailed results found; skipping KPI export"
fi

# Start services in the background. "up -d" returns before they are ready, so
# the helper below polls their health endpoints before the CSV import.
echo "==> Starting VictoriaMetrics and Grafana"
docker compose -f "$repo_root/docker-compose.yml" up -d

# Reusable readiness check: curl --fail treats HTTP errors as failures, while
# --silent keeps retry output quiet. Try 60 times with a 2-second pause (up to
# about two minutes); on timeout, print recent container logs to stderr.
wait_for_service() {
  local url=$1
  local service=$2
  local attempt
  for attempt in {1..60}; do
    if curl --silent --fail "$url" >/dev/null; then
      echo "    $service is ready"
      return 0
    fi
    sleep 2
  done
  echo "Error: $service did not become ready at $url" >&2
  docker compose -f "$repo_root/docker-compose.yml" logs --tail=80 >&2
  return 1
}

wait_for_service "http://127.0.0.1:8428/health" "VictoriaMetrics"
wait_for_service "http://127.0.0.1:3000/api/health" "Grafana"

# POST the CSV to the VictoriaMetric's import endpoint. In the format mapping, each number is a
# one-based CSV column: timestamp, run label, stage label, and metric/value.
# @file tells curl to read the request body from that file.
echo "==> Importing CSV into VictoriaMetrics"
curl --silent --show-error --fail-with-body \
  --request POST \
  --data-binary "@$csv_path" \
  "http://127.0.0.1:8428/api/v1/import/csv?format=1:time:unix_ms,2:label:run,3:label:stage,4:metric:latency_ms"
echo

if [[ -n $trackeval_file && -f $kpi_csv_path ]]; then
  echo "==> Importing TrackEval KPIs into VictoriaMetrics"
  curl --silent --show-error --fail-with-body \
    --request POST \
    --data-binary "@$kpi_csv_path" \
    "http://127.0.0.1:8428/api/v1/import/csv?format=1:time:unix_ms,2:label:run,3:label:tracker,4:label:sequence,5:label:class,6:label:metric,7:metric:trackeval_kpi"
  echo
fi

# URL-encode the run label before placing it in a query parameter, so special
# characters in a custom label cannot break the Grafana link.
encoded_run_id=$(python3 -c 'import sys; from urllib.parse import quote; print(quote(sys.argv[1], safe=""))' "$run_id")
echo "Done."
echo "Run: $run_id"
echo "Grafana dashboard: http://127.0.0.1:3000/d/ros2-latency/ros-2-latency-analysis?var-run=$encoded_run_id"
echo "VictoriaMetrics UI: http://127.0.0.1:8428/vmui"
echo "CSV kept at: $csv_path"
if [[ -n $trackeval_file && -f $kpi_csv_path ]]; then
  echo "TrackEval KPI CSV kept at: $kpi_csv_path"
fi
