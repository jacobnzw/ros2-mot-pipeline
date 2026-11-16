#!/usr/bin/env bash
# Stop on command errors (-e), unset variables (-u), and failed commands inside
# a pipeline (-o pipefail), so an unsuccessful export/import can't look successful.
set -euo pipefail

# Print usage errors to stderr (>&2), keeping normal output separate.
usage() {
  echo "Usage: $0 path/to/recording.rrd [run-id] [--hw hardware] [--precision precision] [--n-threads count] [--seq sequence] [--config config]" >&2
}

# Accept the recording path, an optional run label, and optional metric labels.
if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

# Check required programs before doing any work. command -v returns non-zero if
# a program is unavailable; >/dev/null hides its normal output.
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required to run scripts/eval/extract_pipeline_kpi.py" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Error: Docker with the Compose plugin is required" >&2
  exit 1
fi

# Keep the first argument as the recording path; parse remaining options before
# resolving it so metadata options can appear without a positional run label.
recording_arg=$1
shift
run_id=""
run_id_provided=false
if [[ $# -gt 0 && $1 != --* ]]; then
  run_id=$1
  run_id_provided=true
  shift
fi

hw=""
precision=""
n_threads=""
seq=""
config=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --hw)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      hw=$2
      shift 2
      ;;
    --precision)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      precision=$2
      shift 2
      ;;
    --n-threads)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      n_threads=$2
      shift 2
      ;;
    --seq)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      seq=$2
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      config=$2
      shift 2
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

n_threads_args=()
if [[ -n $n_threads ]]; then
  n_threads_args=(--n-threads "$n_threads")
fi

recording_path=$(realpath -- "$recording_arg")
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
if [[ $run_id_provided == false ]]; then
  if [[ $recording_stem == recording ]]; then
    run_id=$(basename -- "$recording_dir")
  else
    run_id=$recording_stem
  fi
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
uv run --project "$repo_root" python "$repo_root/scripts/eval/extract_pipeline_kpi.py" \
  "$recording_path" --output "$csv_path" --run-id "$run_id" \
  --hw "$hw" --precision "$precision" "${n_threads_args[@]}" --seq "$seq" --config "$config"

# TrackEval is an optional post-processing step. When its detailed result files
# are present in the run folder, export their per-sequence and combined KPIs.
trackeval_file=$(find "$recording_dir" -type f -name '*_detailed.csv' -print -quit)
if [[ -n $trackeval_file ]]; then
  echo "==> Exporting TrackEval KPIs from $recording_dir"
  uv run --project "$repo_root" python "$repo_root/scripts/eval/trackeval_to_vm.py" \
    "$recording_dir" --output "$kpi_csv_path" --run-id "$run_id" \
    --hw "$hw" --precision "$precision" "${n_threads_args[@]}" --config "$config"
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
# one-based CSV column: timestamp, run/name labels, optional metadata labels, and metric/value.
# @file tells curl to read the request body from that file.
echo "==> Importing CSV into VictoriaMetrics"
curl --silent --show-error --fail-with-body \
  --request POST \
  --data-binary "@$csv_path" \
  "http://127.0.0.1:8428/api/v1/import/csv?format=1:time:unix_ms,2:label:run,3:label:name,4:label:hardware,5:label:precision,6:label:n_threads,7:label:rate,8:label:sequence,9:label:config,10:metric:pipeline_kpi"
echo

if [[ -n $trackeval_file && -f $kpi_csv_path ]]; then
  echo "==> Importing TrackEval KPIs into VictoriaMetrics"
  curl --silent --show-error --fail-with-body \
    --request POST \
    --data-binary "@$kpi_csv_path" \
    "http://127.0.0.1:8428/api/v1/import/csv?format=1:time:unix_ms,2:label:hardware,3:label:precision,4:label:n_threads,5:label:rate,6:label:config,7:label:tracker,8:label:sequence,9:label:class,10:label:metric,11:metric:mot_kpi"
  echo
fi

# URL-encode the config value before placing it in a query parameter.
encoded_config=$(python3 -c 'import sys; from urllib.parse import quote; print(quote(sys.argv[1], safe=""))' "$config")
echo "Done."
echo "Run: $run_id"
dashboard_url="http://127.0.0.1:3000/d/ros2-latency/pipeline-details"
if [[ -n $config ]]; then
  dashboard_url+="?var-config=$encoded_config"
fi
echo "Grafana dashboard: $dashboard_url"
echo "VictoriaMetrics UI: http://127.0.0.1:8428/vmui"
echo "CSV kept at: $csv_path"
if [[ -n $trackeval_file && -f $kpi_csv_path ]]; then
  echo "TrackEval KPI CSV kept at: $kpi_csv_path"
fi
