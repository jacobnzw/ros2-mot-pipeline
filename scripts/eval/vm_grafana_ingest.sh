#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <label>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

label=$1
if [[ -z $label || $label == "." || $label == ".." || $label == */* || $label == *\\* ]]; then
  echo "Error: label must be a non-empty directory name without path separators" >&2
  exit 2
fi

for command_name in docker curl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Error: $command_name is required" >&2
    exit 1
  fi
done
if ! docker compose version >/dev/null 2>&1; then
  echo "Error: Docker with the Compose plugin is required" >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
run_dir="$repo_root/runs/$label"
pipeline_dir="$run_dir/eval/pipeline"
tracker_kpi_csv="$run_dir/eval/tracker_kpi.csv"

shopt -s nullglob
pipeline_csvs=("$pipeline_dir"/kpi_*.csv)
shopt -u nullglob

if [[ ! -d $run_dir ]]; then
  echo "Error: run directory does not exist: $run_dir" >&2
  exit 1
fi
if [[ ${#pipeline_csvs[@]} -eq 0 ]]; then
  echo "Error: no pipeline KPI CSVs found in $pipeline_dir" >&2
  exit 1
fi
if [[ ! -f $tracker_kpi_csv ]]; then
  echo "Error: tracker KPI CSV does not exist: $tracker_kpi_csv" >&2
  exit 1
fi

compose_file="$repo_root/docker-compose.yml"
vm_url="http://127.0.0.1:8428"

echo "==> Starting VictoriaMetrics and Grafana"
docker compose -f "$compose_file" up -d

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
  docker compose -f "$compose_file" logs --tail=80 >&2
  return 1
}

wait_for_service "$vm_url/health" "VictoriaMetrics"
wait_for_service "http://127.0.0.1:3000/api/health" "Grafana"

import_csv() {
  local csv_path=$1
  local format=$2
  echo "==> Importing $csv_path"
  curl --silent --show-error --fail-with-body \
    --request POST \
    --data-binary "@$csv_path" \
    "$vm_url/api/v1/import/csv?format=$format"
  echo
}

pipeline_format="1:time:unix_ms,2:label:run,3:label:stage,4:label:hardware,5:label:precision,6:label:sequence,7:label:label,8:metric:pipeline_kpi"
tracker_format="1:time:unix_ms,2:label:hardware,3:label:precision,4:label:label,5:label:tracker,6:label:sequence,7:label:class,8:label:metric,9:metric:mot_kpi"

for csv_path in "${pipeline_csvs[@]}"; do
  import_csv "$csv_path" "$pipeline_format"
done
import_csv "$tracker_kpi_csv" "$tracker_format"

echo "Done importing run '$label'."
echo "Grafana: http://127.0.0.1:3000"
echo "VictoriaMetrics: http://127.0.0.1:8428/vmui"
