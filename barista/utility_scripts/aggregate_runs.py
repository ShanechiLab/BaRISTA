import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

KEY = 'TEST'  # Options: 'VAL', 'TEST', 'LAST_TEST'

def parse_summary(path):
    try:
        txt = open(path).read()
        mean = float(re.search(rf"{KEY}_MEAN=([0-9.]+)", txt).group(1))
        std = float(re.search(rf"{KEY}_STD=([0-9.]+)", txt).group(1))
        ckpt_line = re.search(r"Checkpoint:\s*(.*)", txt).group(1)
        model = os.path.basename(ckpt_line).replace(".ckpt", "")
        return model, f"{mean:.3f} ± {std:.3f}"
    except:
        return None

def parse_from_seeds(folder):
    logs = sorted(glob.glob(os.path.join(folder, "seed_*.log")))
    expected_seeds = 5

    if not logs:
        print(f"WARNING: No seed logs found in {folder}")
        return None

    auc_pattern = r"TEST AUC:\s*([0-9.]+)" if KEY == "TEST" else \
                  r"LAST TEST AUC:\s*([0-9.]+)" if KEY == "LAST_TEST" else None
    if auc_pattern is None:
        return None

    ckpt_pattern = r"'checkpoint_path':\s*'([^']*)'"

    vals, model_name, valid_logs = [], None, 0

    for log in logs:
        try:
            txt = open(log).read()
            m = re.search(auc_pattern, txt)
            if m:
                vals.append(float(m.group(1)))
                valid_logs += 1

            cm = re.search(ckpt_pattern, txt)
            if cm:
                ckpt_path = cm.group(1)
                model_name = os.path.basename(ckpt_path).replace(".ckpt", "")
        except:
            pass

    model_name = model_name or "unknown"
    if model_name == '':
        model_name = "random"

    if valid_logs != expected_seeds and model_name != 'random':
        print(f"WARNING: Incomplete seeds for {model_name} in {folder} "
              f"(found {valid_logs}/{expected_seeds})")

    if not vals:
        return None

    mean, std = float(np.mean(vals)), float(np.std(vals))
    return model_name, f"{mean:.3f} ± {std:.3f}"

def parse_summary_or_seeds(folder):
    summary_path = os.path.join(folder, "summary.txt")
    if os.path.exists(summary_path):
        parsed = parse_summary(summary_path)
        if parsed:
            return parsed
    return parse_from_seeds(folder)

def extract_mean(x):
    if isinstance(x, str) and "±" in x:
        return float(x.split("±")[0].strip())
    return np.nan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results folder")
    args = parser.parse_args()
    ROOT = args.results_dir

    rows, subjects, tasks, models, folds = [], set(), set(), set(), set()

    # Collect data from folders
    for folder in os.listdir(ROOT):
        fpath = os.path.join(ROOT, folder)
        if not os.path.isdir(fpath):
            continue

        parts = folder.split("_")
        if len(parts) < 6:
            continue

        subj = parts[1]
        task = parts[4]
        if len(parts) > 5 and parts[5] in ["onset", "vs", "nonspeech", "speech", "time"]:
            task += f"_{parts[5]}"
        if len(parts) > 6 and parts[6] == "nonspeech":
            task += f"_{parts[6]}"

        fold = None
        for p in parts:
            if p.startswith("fold"):
                fold = int(p.replace("fold", ""))
                folds.add(fold)
                break

        parsed = parse_summary_or_seeds(fpath)
        if not parsed:
            continue

        model, value = parsed
        subjects.add(subj)
        tasks.add(task)
        models.add(model)
        rows.append((task, model, subj, fold, value))

    # Build DataFrame
    subjects = sorted(subjects, key=lambda x: int(x))
    df = pd.DataFrame(columns=["task", "model", "fold"] + subjects)

    for task in sorted(tasks):
        for model in sorted(models):
            all_folds = sorted(folds) + [None]
            for fold in all_folds:
                subset = [(s, v) for t, m, s, f, v in rows if t == task and m == model and f == fold]
                if not subset:
                    continue
                row = {"task": task, "model": model, "fold": fold if fold is not None else ""}
                for subj, val in subset:
                    row[subj] = val
                df.loc[len(df)] = row

    # Add AVG column
    subj_cols = [c for c in df.columns if c not in ["task", "model", "fold"]]
    df["avg"] = df[subj_cols].applymap(extract_mean).mean(axis=1)
    df["avg"] = df["avg"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    # Add final AVG rows per (task, model)
    avg_rows = []
    for (task, model), group in df.groupby(["task", "model"]):
        subj_avgs = {}
        for subj in subj_cols:
            vals = [float(v.split("±")[0].strip()) for v in group[subj] if isinstance(v, str) and "±" in v]
            subj_avgs[subj] = f"{np.mean(vals):.3f}" if vals else ""
        overall_vals = [float(v) for v in subj_avgs.values() if v != ""]
        overall_avg = f"{np.mean(overall_vals):.3f}" if overall_vals else ""
        row = {"task": task, "model": model, "fold": "AVG", "avg": overall_avg}
        row.update(subj_avgs)
        avg_rows.append(row)

    df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
