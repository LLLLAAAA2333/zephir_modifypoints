from collections import Counter
import h5py
import numpy as np
import argparse

def _build_parser():
    parser = argparse.ArgumentParser(description="Summarize per-frame worldline anomalies.")
    parser.add_argument("--annotations_path", help="Path to annotations.h5")
    return parser

def summarize_worldlines(annotations_path):
    """
    Check per-frame worldline anomalies: missing, extra, duplicates.
    Infers expected_ids from all unique worldline_id values in the file.
    """
    with h5py.File(annotations_path, "r") as f:
        t_idx = f["/t_idx"][:]
        worldline = f["/worldline_id"][:]
    
    expected_ids = set(np.unique(worldline))
    expected_count = len(expected_ids)
    issues = {}

    for t in np.unique(t_idx):
        mask = t_idx == t
        ids = worldline[mask]
        counts = Counter(ids)
        observed = set(ids)

        missing = sorted(expected_ids - observed)
        extra = sorted(observed - expected_ids)
        duplicates = {wl: c for wl, c in counts.items() if c > 1}

        if missing or extra or duplicates:
            issues[int(t)] = {
                "missing": missing,
                "extra": extra,
                "duplicates": duplicates,
            }
    
    # Print summary
    if not issues:
        print(f"✓ All frames contain the expected {expected_count} worldlines.")
    else:
        print(f"Found {len(issues)} frames with anomalies (expected {expected_count} worldlines):\n")
        for t in sorted(issues.keys()):
            issue = issues[t]
            print(f"Frame t={t}:")
            if issue["missing"]:
                print(f"  ✗ Missing:     {issue['missing']}")
            if issue["extra"]:
                print(f"  ✗ Extra:       {issue['extra']}")
            if issue["duplicates"]:
                for wl, count in sorted(issue["duplicates"].items()):
                    print(f"  ⚠ Duplicate:   worldline_id {wl} appears {count} times")
            print()
    
    return issues

if __name__ == "__main__":
    # annotations_path = r"I:\WJH\infer\manual\registration_annotation\20250730\w3_freelymoving\w3_manual\w3\vol_0_99\annotations.h5"
    # summarize_worldlines(annotations_path)
    parser = _build_parser()
    args = parser.parse_args()
    summarize_worldlines(args.annotations_path)
