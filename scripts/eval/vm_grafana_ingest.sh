#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--ingest-only] <config-or-glob>" >&2
  echo "  <config-or-glob>  runs/ directory name or quoted glob pattern, e.g. 'rtx3090*'" >&2
  echo "  --ingest-only     skip starting the Compose stack; only POST CSVs to a running VictoriaMetrics" >&2
}

ingest_only=false
config=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --ingest-only)
      ingest_only=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -n $config ]]; then
        echo "Error: expected a single config or quoted glob pattern; quote patterns like 'rtx3090*'" >&2
        usage
        exit 2
      fi
      config=$1
      shift
      ;;
  esac
done

if [[ -z $config || $config == "." || $config == ".." || $config == */* || $config == *\\* ]]; then
  echo "Error: config must be a non-empty directory name or glob pattern without path separators" >&2
  usage
  exit 2
fi

required_commands=(curl)
if [[ $ingest_only == false ]]; then
  required_commands+=(docker)
fi
for command_name in "${required_commands[@]}"; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Error: $command_name is required" >&2
    exit 1
  fi
done
if [[ $ingest_only == false ]] && ! docker compose version >/dev/null 2>&1; then
  echo "Error: Docker with the Compose plugin is required" >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
runs_dir="$repo_root/runs"

# Match directory names against the pattern directly, so patterns with spaces or
# special characters are not subject to word splitting.
matched_configs=()
shopt -s nullglob
for dir in "$runs_dir"/*/; do
  name=$(basename -- "$dir")
  # shellcheck disable=SC2053
  if [[ $name == $config ]]; then
    matched_configs+=("$name")
  fi
done
shopt -u nullglob

if [[ ${#matched_configs[@]} -eq 0 ]]; then
  echo "Error: no directories in $runs_dir match '$config'" >&2
  exit 1
fi

# Keep only configs that have the evaluation outputs; skipped ones are reported.
ingestible_configs=()
for name in "${matched_configs[@]}"; do
  config_dir="$runs_dir/$name"
  shopt -s nullglob
  pipeline_csvs=("$config_dir"/eval/pipeline/kpi_*.csv)
  shopt -u nullglob
  if [[ ${#pipeline_csvs[@]} -eq 0 ]]; then
    echo "Skipping '$name': no pipeline KPI CSVs found in $config_dir/eval/pipeline" >&2
  elif [[ ! -f $config_dir/eval/tracker_kpi.csv ]]; then
    echo "Skipping '$name': tracker KPI CSV does not exist: $config_dir/eval/tracker_kpi.csv" >&2
  else
    ingestible_configs+=("$name")
  fi
done

if [[ ${#ingestible_configs[@]} -eq 0 ]]; then
  echo "Error: no matching config has the required KPI CSVs" >&2
  exit 1
fi

compose_file="$repo_root/docker-compose.yml"
vm_url="http://127.0.0.1:8428"

if [[ $ingest_only == false ]]; then
  echo "==> Starting VictoriaMetrics and Grafana"
  docker compose -f "$compose_file" up -d
fi

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
  if [[ $ingest_only == false ]]; then
    docker compose -f "$compose_file" logs --tail=80 >&2
  fi
  return 1
}

if [[ $ingest_only == true ]]; then
  if ! curl --silent --fail "$vm_url/health" >/dev/null; then
    echo "Error: VictoriaMetrics is not reachable at $vm_url; start it or omit --ingest-only" >&2
    exit 1
  fi
else
  wait_for_service "$vm_url/health" "VictoriaMetrics"
  wait_for_service "http://127.0.0.1:3000/api/health" "Grafana"
fi

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

pipeline_format="1:time:unix_ms,2:label:run,3:label:name,4:label:hardware,5:label:precision,6:label:n_threads,7:label:rate,8:label:sequence,9:label:config,10:metric:pipeline_kpi"
tracker_format="1:time:unix_ms,2:label:hardware,3:label:precision,4:label:n_threads,5:label:rate,6:label:config,7:label:tracker,8:label:sequence,9:label:class,10:label:metric,11:metric:mot_kpi"
device_metrics_format="1:time:unix_ms,2:metric:cpu_temp_c,3:metric:cpu_usage_pct,4:metric:gpu_temp_c,5:metric:gpu_usage_pct,6:metric:gpu_mem_used_mb,7:metric:gpu_mem_total_mb,8:metric:aux_temp_c,9:metric:emc_usage_pct,10:metric:ram_used_mb,11:metric:ram_total_mb,12:metric:swap_used_mb,13:metric:power_mw,14:metric:fan_rpm,15:label:config,16:label:hardware"

for name in "${ingestible_configs[@]}"; do
  config_dir="$runs_dir/$name"
  tracker_kpi_csv="$config_dir/eval/tracker_kpi.csv"
  device_metrics_csv="$config_dir/eval/device_metrics.csv"

  echo "==> Config '$name'"
  shopt -s nullglob
  pipeline_csvs=("$config_dir"/eval/pipeline/kpi_*.csv)
  shopt -u nullglob
  for csv_path in "${pipeline_csvs[@]}"; do
    import_csv "$csv_path" "$pipeline_format"
  done
  import_csv "$tracker_kpi_csv" "$tracker_format"
  if [[ -f $device_metrics_csv ]]; then
    import_csv "$device_metrics_csv" "$device_metrics_format"
  else
    echo "Skipping device metrics; CSV not found: $device_metrics_csv"
  fi
done

echo "Done importing ${#ingestible_configs[@]} config(s): ${ingestible_configs[*]}"
echo "Grafana: http://127.0.0.1:3000"
echo "VictoriaMetrics: http://127.0.0.1:8428/vmui"