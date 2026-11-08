"""Compare CAIDE v17.1.0 with published TensorRT-LLM static-batch measurements."""
import argparse
import csv
import math
import statistics
from pathlib import Path

from caide import DeploymentState, Observation, __version__, evaluate_request
from caide.catalog import get_hardware, get_model

# Fitted on the four observations in docs/model.md section 8.
# Keep this fixed for the external holdout. Prefill scale stays at 1.
DECODE_SCALE = 0.7404131173250055


def predict(observation, decode_scale):
    state = DeploymentState(
        observation.model, observation.hardware, observation.serving(1.0, decode_scale)
    )
    performance = evaluate_request(
        state, observation.workload(), batch_override=observation.batch
    )
    if not performance.feasible:
        return "infeasible", performance.batch, None
    if performance.batch_truncated:
        return "batch_truncated", performance.batch, None
    return "matched", performance.batch, performance.throughput_qps * observation.tokens_out


def compare(row):
    device_count = int(row["tensor_parallel"])
    measured = int(row["reported_output_tps_per_gpu"]) * device_count
    model = get_model(row["caide_model"]).with_precision(
        float(row["caide_weight_bytes"]), float(row["caide_kv_bytes"])
    )
    observation = Observation(
        model=model,
        hardware=get_hardware(row["caide_hardware"]),
        n_accelerators=device_count,
        batch=int(row["reported_batch"]),
        tokens_in=int(row["input_tokens"]),
        tokens_out=int(row["output_tokens"]),
        measured_output_tps=measured,
        precision=row["reported_precision"].lower(),
        source=row["source_url"],
        convention="aggregate",
    )
    status, batch, baseline = predict(observation, 1.0)
    fixed_status, fixed_batch, calibrated = predict(observation, DECODE_SCALE)
    if (status, batch) != (fixed_status, fixed_batch):
        raise RuntimeError(f"Capacity changed with decode calibration for {row['id']}")
    return {
        "id": row["id"],
        "reported_replica_output_tps": measured,
        "reported_batch": observation.batch,
        "caide_batch": batch,
        "status": status,
        "uncalibrated_replica_output_tps": baseline,
        "fixed_calibration_replica_output_tps": calibrated,
        "uncalibrated_ratio": baseline / measured if baseline is not None else None,
        "fixed_calibration_ratio": calibrated / measured if calibrated is not None else None,
    }


def report(results):
    matched = [row for row in results if row["status"] == "matched"]
    print(f"Source rows: {len(results)}. Batch-matched: {len(matched)}.")
    for status in ("batch_truncated", "infeasible"):
        print(f"{status}: {sum(row['status'] == status for row in results)}.")
    for field in ("uncalibrated_ratio", "fixed_calibration_ratio"):
        ratios = [row[field] for row in matched]
        if not ratios:
            continue
        within = sum(0.5 <= ratio <= 2.0 for ratio in ratios)
        log_rmse = math.sqrt(statistics.mean(math.log(ratio) ** 2 for ratio in ratios))
        over = sum(ratio > 1.0 for ratio in ratios)
        under = sum(ratio < 1.0 for ratio in ratios)
        print(f"{field}: log RMSE {log_rmse:.6f}; within 2x {within}/{len(ratios)}; "
              f"over {over}; under {under}.")


def main():
    if __version__ != "17.1.0":
        raise SystemExit(f"Install the archived CAIDE v17.1.0 release. Found {__version__}.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = Path(__file__).with_name("holdout_inputs.csv")
    with inputs.open(encoding="utf-8", newline="") as stream:
        results = [compare(row) for row in csv.DictReader(stream)]
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    report(results)


if __name__ == "__main__":
    main()
