"""Log device temperatures and resource usage to CSV."""

import argparse
import csv
import math
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, TextIO

import psutil

FIELDS = [
    "timestamp_ms",
    "cpu_temp_c",
    "cpu_usage_pct",
    "gpu_temp_c",
    "gpu_usage_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "aux_temp_c",
    "emc_usage_pct",
    "ram_used_mb",
    "ram_total_mb",
    "swap_used_mb",
    "power_mw",
    "fan_rpm",
    "config",
    "hardware",
]

MEGABYTES = 1024**-2


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def get_cpu_temp() -> float | str:
    """Read the first available CPU thermal zone, or return "nan" if unavailable."""
    temps = psutil.sensors_temperatures()
    for key in ("coretemp", "k10temp", "cpu_thermal"):
        if key in temps and temps[key]:
            return temps[key][0].current
    return "nan"


def get_gpu_stats() -> tuple[float | None, float | None, float | None, float | None]:
    """Query NVIDIA GPU 0; return empty values when nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        values = [float(value) for value in result.stdout.strip().split(",")]
        if len(values) != 4:
            return None, None, None, None
        return values[0], values[1], values[2], values[3]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None, None, None


def get_pc_stats() -> dict[str, float | str | None]:
    gpu_temp, gpu_usage, gpu_mem_used, gpu_mem_total = get_gpu_stats()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "cpu_temp_c": get_cpu_temp(),
        "cpu_usage_pct": psutil.cpu_percent(interval=None),
        "gpu_temp_c": gpu_temp,
        "gpu_usage_pct": gpu_usage,
        "gpu_mem_used_mb": gpu_mem_used,
        "gpu_mem_total_mb": gpu_mem_total,
        "aux_temp_c": None,
        "emc_usage_pct": None,
        "ram_used_mb": round(memory.used * MEGABYTES, 1),
        "ram_total_mb": round(memory.total * MEGABYTES, 1),
        "swap_used_mb": round(swap.used * MEGABYTES, 1),
        "power_mw": None,
        "fan_rpm": None,
    }


def get_jetson_stats(stats: dict) -> dict[str, float | str | None]:
    return {
        "cpu_temp_c": stats.get("cpu_temp"),
        "cpu_usage_pct": stats.get("cpu_usage"),
        "gpu_temp_c": stats.get("gpu_temp"),
        "gpu_usage_pct": stats.get("gpu_usage"),
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "aux_temp_c": stats.get("aux_temp"),
        "emc_usage_pct": stats.get("emc_usage"),
        "ram_used_mb": stats.get("ram_used"),
        "ram_total_mb": stats.get("ram_total"),
        "swap_used_mb": stats.get("swap_used"),
        "power_mw": stats.get("power"),
        "fan_rpm": stats.get("fan_rpm"),
    }


def make_sample_row(
    config: str,
    hardware: str,
    stats: dict[str, float | str | None],
    timestamp_ms: int | None = None,
) -> dict[str, int | float | str]:
    return {
        "timestamp_ms": time.time_ns() // 1_000_000 if timestamp_ms is None else timestamp_ms,
        **{field: "nan" if value is None else value for field, value in stats.items()},
        "config": config,
        "hardware": hardware,
    }


def write_samples(
    writer: csv.DictWriter,
    output_file: TextIO,
    args: argparse.Namespace,
    get_stats: Callable[[], dict[str, float | str | None]],
) -> None:
    while True:
        writer.writerow(make_sample_row(args.config, args.hardware, get_stats()))
        output_file.flush()
        time.sleep(1 / args.freq)


def main(args: argparse.Namespace) -> None:
    psutil.cpu_percent(interval=None)
    output_path = Path(f"runs/{args.config}/eval/{args.out_name}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        print(f"Logging {args.hardware} metrics to {output_path} ...", flush=True)

        if args.hardware == "jetson":
            try:
                from jtop import jtop
            except ImportError as error:
                raise RuntimeError("Jetson monitoring requires the 'jtop' package") from error
            with jtop() as jetson:
                write_samples(writer, output_file, args, lambda: get_jetson_stats(jetson.stats))
        else:
            write_samples(writer, output_file, args, get_pc_stats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log device temperatures and resource usage to CSV.")
    parser.add_argument("--hardware", choices=("pc", "jetson"), default="pc", help="Device type to monitor.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config label identifying the experiment. Output is written under runs/{config}/eval.",
    )
    parser.add_argument("--out-name", type=str, default="device_metrics", help="Output CSV file name.")
    parser.add_argument("--freq", type=positive_float, default=1.0, help="Logging frequency in Hz.")
    return parser


def handle_sigterm(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        main(build_parser().parse_args())
    except KeyboardInterrupt:
        print("\nStopped.")
