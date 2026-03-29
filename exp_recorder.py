#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Dict, List, Tuple, OrderedDict


EPOCH_LINE_PREFIX = "Epoch(val)"


def parse_epoch_and_metrics_ordered(line: str) -> Tuple[int, List[Tuple[str, float]]]:
    """
    Parse a validation summary line for epoch number and an ordered list of (metric, value).

    Example line:
    "Epoch(val) [4][61/61]    AUC: 0.9822  ...  mAUC^0.1: 0.3236  AP: 0.9959 ...  data_time: 0.0272  time: 0.1450"

    We keep the original order of metrics as they appear in the log.
    """
    if EPOCH_LINE_PREFIX not in line:
        return -1, []

    epoch_match = re.search(r"Epoch\(val\)\s*\[(\d+)\]", line)
    if not epoch_match:
        return -1, []
    epoch = int(epoch_match.group(1))

    # Only parse the substring after the epoch metadata: Epoch(val) [E][X/Y]    ...
    tail_match = re.search(r"Epoch\(val\)\s*\[\d+\]\[\d+/\d+\]\s*(.*)$", line)
    tail = tail_match.group(1) if tail_match else line

    ordered_metrics: List[Tuple[str, float]] = []
    for key, value in re.findall(r"([A-Za-z0-9_.^]+):\s*([-+]?(?:\d+\.\d+|\d+))", tail):
        if key in ("data_time", "time"):
            continue
        try:
            ordered_metrics.append((key, float(value)))
        except ValueError:
            continue

    return epoch, ordered_metrics


def scan_log_for_epochs(log_path: str) -> List[Tuple[int, List[Tuple[str, float]]]]:
    results: List[Tuple[int, List[Tuple[str, float]]]] = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if EPOCH_LINE_PREFIX not in line:
                continue
            epoch, ordered_metrics = parse_epoch_and_metrics_ordered(line)
            if epoch >= 0 and ordered_metrics:
                results.append((epoch, ordered_metrics))
    if not results:
        raise ValueError("未能在日志中找到包含验证指标的行。")
    return results


def metrics_to_dict(ordered_metrics: List[Tuple[str, float]]) -> Dict[str, float]:
    return {k: v for k, v in ordered_metrics}


def pick_best_epoch(entries: List[Tuple[int, List[Tuple[str, float]]]], target_metric: str) -> Tuple[int, List[Tuple[str, float]]]:
    best_epoch = -1
    best_metrics: List[Tuple[str, float]] = []
    best_value = float('-inf')
    for epoch, ordered_metrics in entries:
        m = metrics_to_dict(ordered_metrics)
        if target_metric not in m:
            continue
        if m[target_metric] > best_value:
            best_value = m[target_metric]
            best_epoch = epoch
            best_metrics = ordered_metrics
    if best_epoch < 0:
        raise ValueError(f"日志中未找到目标指标 {target_metric} 。")
    return best_epoch, best_metrics


def ensure_csv_append_rows(csv_path: str, header_fields: List[str], rows: List[List[str]]) -> None:
    """
    Ensure CSV exists with the given header. If exists and header differs, rewrite with union while
    preserving provided header order (exp_id, exp_dir, epoch, metrics..., comment) and appending any
    missing fields at the end in encountered order.
    """
    existing_rows: List[Dict[str, str]] = []
    existing_header: List[str] = []

    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as rf:
            reader = csv.reader(rf)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []
            for r in reader:
                existing_rows.append({h: (r[i] if i < len(r) else "") for i, h in enumerate(existing_header)})

    # Build final header: respect provided header order; if existing has more, append them (excluding duplicates)
    final_header: List[str] = []
    seen = set()
    for h in header_fields:
        if h not in seen:
            final_header.append(h)
            seen.add(h)
    for h in existing_header:
        if h not in seen:
            final_header.append(h)
            seen.add(h)

    # Normalize existing rows to final header
    normalized_existing: List[List[str]] = []
    for r in existing_rows:
        normalized_existing.append([r.get(h, "") for h in final_header])

    # Normalize new rows to final header
    # First, map header to index for current header_fields
    header_index = {h: i for i, h in enumerate(header_fields)}
    # Extend any missing columns for rows to match final header length
    normalized_new: List[List[str]] = []
    for r in rows:
        # r is aligned with header_fields
        base = {h: (r[header_index[h]] if h in header_index and header_index[h] < len(r) else "") for h in header_fields}
        normalized_new.append([base.get(h, "") for h in final_header])

    with open(csv_path, 'w', newline='', encoding='utf-8') as wf:
        writer = csv.writer(wf)
        writer.writerow(final_header)
        for row_vals in normalized_existing:
            writer.writerow(row_vals)
        for row_vals in normalized_new:
            writer.writerow(row_vals)


def resolve_log_path(input_path: str) -> str:
    """Resolve -l/--log_path which can be a file, a directory, a bare exp name like
    '<config_name>_YYYYMMDD_HHMMSS', or a bare timestamp 'YYYYMMDD_HHMMSS'. Returns absolute file path.
    """
    project_root = os.path.abspath(os.path.dirname(__file__))

    def abs_path(p: str) -> str:
        if not os.path.isabs(p):
            p = os.path.abspath(os.path.join(os.getcwd(), p))
        return p

    # Direct file path
    p = abs_path(input_path)
    if os.path.isfile(p):
        return p

    # Directory path -> infer {basename}.log inside it
    if os.path.isdir(p):
        base = os.path.basename(p.rstrip(os.sep))
        candidate = os.path.join(p, f"{base}.log")
        if os.path.isfile(candidate):
            return candidate

    # Bare exp name like <config_name>_YYYYMMDD_HHMMSS (e.g., top_cap_20251016_172904, top_cap_cnn_...)
    # Prefer exact subdir by prefix
    m = re.match(r"^([A-Za-z0-9_]+)_(\d{8}_\d{6})$", input_path)
    if m:
        subgroup = m.group(1)
        ts = m.group(2)
        candidate = os.path.join(project_root, "work_dirs", subgroup, ts, f"{ts}.log")
        if os.path.isfile(candidate):
            return candidate
        # fallback: try codes/<subgroup>/<ts>/<ts>.log
        candidate_codes = os.path.join(project_root, "codes", subgroup, ts, f"{ts}.log")
        if os.path.isfile(candidate_codes):
            return candidate_codes

    # Bare timestamp like YYYYMMDD_HHMMSS -> scan all subdirs under work_dirs and codes
    if re.match(r"^\d{8}_\d{6}$", input_path):
        ts = input_path
        for base_dir in (os.path.join(project_root, "work_dirs"), os.path.join(project_root, "codes")):
            if os.path.isdir(base_dir):
                try:
                    for subgroup in os.listdir(base_dir):
                        subgroup_path = os.path.join(base_dir, subgroup)
                        if not os.path.isdir(subgroup_path):
                            continue
                        candidate = os.path.join(subgroup_path, ts, f"{ts}.log")
                        if os.path.isfile(candidate):
                            return candidate
                except OSError:
                    pass

    # Relative to project root as file
    proj_rel = os.path.join(project_root, input_path)
    if os.path.isfile(proj_rel):
        return proj_rel
    # Relative to project root as directory -> infer {basename}.log
    if os.path.isdir(proj_rel):
        base = os.path.basename(proj_rel.rstrip(os.sep))
        candidate = os.path.join(proj_rel, f"{base}.log")
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"无法解析日志路径: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or delete experiment records in records.csv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-a", "--add", action="store_true", help="Add mode (default)")
    mode.add_argument("-d", "--delete", action="store_true", help="Delete mode (by exp_id)")
    mode.add_argument("-m", "--modify", action="store_true", help="Modify mode (by exp_id)")
    parser.add_argument("-l", "--log_path", required=False, help="Path to log file, e.g., work_dirs/top_cap/.../*.log")
    parser.add_argument("-c", "--comment", required=False, default="", help="Optional experiment comment when adding")
    parser.add_argument("-i", "--exp_id", type=int, required=False, help="Experiment id (override when adding; selector when deleting)")
    args = parser.parse_args()

    records_csv = "/home/zhaotianhao/Code/TOP/records.csv"

    # Determine mode (default add)
    is_delete = bool(args.delete)
    is_modify = bool(args.modify)
    # If neither flag provided, default to add
    if not args.add and not args.delete and not args.modify:
        is_delete = False
        is_modify = False

    # Delete mode: delete by exp_id (-i)
    if is_delete:
        if args.exp_id is None:
            raise SystemExit("删除模式必须提供 -i/--exp_id")
        if not os.path.exists(records_csv):
            return
        with open(records_csv, 'r', newline='', encoding='utf-8') as rf:
            reader = csv.reader(rf)
            try:
                header = next(reader)
            except StopIteration:
                return
            exp_id_idx = header.index("exp_id") if "exp_id" in header else -1
            kept_rows: List[List[str]] = [header]
            target_id = args.exp_id
            for row in reader:
                drop = False
                if target_id is not None and exp_id_idx >= 0 and len(row) > exp_id_idx:
                    try:
                        rid = int(row[exp_id_idx])
                        if rid == target_id:
                            drop = True
                    except ValueError:
                        pass
                if not drop:
                    kept_rows.append(row)
        with open(records_csv, 'w', newline='', encoding='utf-8') as wf:
            writer = csv.writer(wf)
            writer.writerows(kept_rows)
        return

    # Modify mode: modify rows by exp_id (-i). Currently supports updating comment (-c)
    if is_modify:
        if args.exp_id is None:
            raise SystemExit("修改模式必须提供 -i/--exp_id")
        if not os.path.exists(records_csv):
            return
        with open(records_csv, 'r', newline='', encoding='utf-8') as rf:
            reader = csv.reader(rf)
            try:
                header = next(reader)
            except StopIteration:
                return
            try:
                exp_id_idx = header.index("exp_id")
            except ValueError:
                raise SystemExit("records.csv 缺少 exp_id 列，无法进行修改")
            try:
                comment_idx = header.index("comment")
            except ValueError:
                # if no comment column yet, append one to header
                header.append("comment")
                comment_idx = len(header) - 1
            updated: List[List[str]] = [header]
            for row in reader:
                # ensure row length
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                try:
                    rid = int(row[exp_id_idx])
                except ValueError:
                    updated.append(row)
                    continue
                if rid == args.exp_id:
                    if args.comment is not None:
                        row[comment_idx] = args.comment
                updated.append(row)
        with open(records_csv, 'w', newline='', encoding='utf-8') as wf:
            writer = csv.writer(wf)
            writer.writerows(updated)
        return

    # Add mode requires log_path
    if not args.log_path:
        raise SystemExit("增加模式必须提供 -l/--log_path")
    log_path = resolve_log_path(args.log_path)

    entries = scan_log_for_epochs(log_path)

    # Determine exp_id by counting group occurrences in CSV (groups are contiguous rows with same exp_id)
    next_exp_id = 1
    if args.exp_id is not None:
        next_exp_id = int(args.exp_id)
    else:
        if os.path.exists(records_csv):
            with open(records_csv, 'r', newline='', encoding='utf-8') as rf:
                reader = csv.reader(rf)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
                exp_id_idx = header.index("exp_id") if "exp_id" in header else -1
                if exp_id_idx >= 0:
                    last_id = 0
                    for row in reader:
                        if len(row) > exp_id_idx and row[exp_id_idx].strip().isdigit():
                            last_id = max(last_id, int(row[exp_id_idx].strip()))
                    next_exp_id = last_id + 1

    # Use project root (directory of this script) as base for relative exp_dir
    project_root = os.path.abspath(os.path.dirname(__file__))
    exp_dir = os.path.relpath(os.path.dirname(log_path), start=project_root)

    # Pick best epochs
    best_AUC_epoch, best_AUC_metrics = pick_best_epoch(entries, "AUC")
    best_mAUC_epoch, best_mAUC_metrics = pick_best_epoch(entries, "mAUC")
    best_AUC01_epoch, best_AUC01_metrics = pick_best_epoch(entries, "AUC^0.1")
    best_mAUC01_epoch, best_mAUC01_metrics = pick_best_epoch(entries, "mAUC^0.1")
    best_TTA01_epoch, best_TTA01_metrics = pick_best_epoch(entries, "TTA^0.1")

    # Build ordered metric names as they appear in log for each selected epoch
    # Use the union order from each chosen line; since we write separate rows, we can keep per-row order
    def build_row(exp_id: int, exp_dir: str, epoch: int, ordered_metrics: List[Tuple[str, float]], comment: str) -> Tuple[List[str], List[str]]:
        metric_names = [name for name, _ in ordered_metrics]
        values = [f"{val:.4f}" for _, val in ordered_metrics]
        header = ["exp_id", "exp_dir", "epoch"] + metric_names + ["comment"]
        row = [str(exp_id), exp_dir, str(epoch)] + values + [comment]
        return header, row

    header_a, row_a = build_row(next_exp_id, exp_dir, best_AUC_epoch, best_AUC_metrics, args.comment)
    header_b, row_b = build_row(next_exp_id, exp_dir, best_mAUC_epoch, best_mAUC_metrics, args.comment)
    header_c, row_c = build_row(next_exp_id, exp_dir, best_AUC01_epoch, best_AUC01_metrics, args.comment)
    header_d, row_d = build_row(next_exp_id, exp_dir, best_mAUC01_epoch, best_mAUC01_metrics, args.comment)
    header_e, row_e = build_row(next_exp_id, exp_dir, best_TTA01_epoch, best_TTA01_metrics, args.comment)

    # Build a superset header that can accommodate all five rows while preserving relative metric order per first occurrence
    combined_header: List[str] = []
    def extend_header(h: List[str]):
        for col in h:
            if col not in combined_header:
                combined_header.append(col)

    extend_header(header_a)
    extend_header(header_b)
    extend_header(header_c)
    extend_header(header_d)
    extend_header(header_e)

    # Align rows to the combined header
    def align_row(header: List[str], row: List[str], final_header: List[str]) -> List[str]:
        idx = {h: i for i, h in enumerate(header)}
        return [row[idx[col]] if col in idx else "" for col in final_header]

    final_rows = [
        align_row(header_a, row_a, combined_header),
        align_row(header_b, row_b, combined_header),
        align_row(header_c, row_c, combined_header),
        align_row(header_d, row_d, combined_header),
        align_row(header_e, row_e, combined_header),
    ]

    ensure_csv_append_rows(records_csv, combined_header, final_rows)


if __name__ == "__main__":
    main()


