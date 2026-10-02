import csv
import numpy as np
from collections import defaultdict

input_file = "Results/results.csv"
output_file = "Results/summary_results.csv"

# group results by dataset and hidden neurons
groups = defaultdict(list)

with open(input_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["dataset"], int(row["hidden"]))
        groups[key].append(row)

# columns for summary
fields = [
    "dataset",
    "hidden",
    "runs",
    "train_mean",
    "train_std",
    "val_mean",
    "val_std",
    "test_mean",
    "test_std",
    "best_epoch_mean"
]

# create summary csv
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for (dataset, hidden), rows in sorted(groups.items()):

        # get values from all runs
        train = np.array([float(r["train_acc"]) for r in rows])
        val = np.array([float(r["val_acc"]) for r in rows])
        test = np.array([float(r["test_acc"]) for r in rows])
        epochs = np.array([float(r["best_epoch"]) for r in rows])

        # calculate vals for summ
        summary = {
            "dataset": dataset,
            "hidden": hidden,
            "runs": len(rows),
            "train_mean": round(train.mean(), 3),
            "train_std": round(train.std(ddof=1), 3),
            "val_mean": round(val.mean(), 3),
            "val_std": round(val.std(ddof=1), 3),
            "test_mean": round(test.mean(), 3),
            "test_std": round(test.std(ddof=1), 3),
            "best_epoch_mean": round(epochs.mean(), 1)
        }

        writer.writerow(summary)

print(f"saved summary to {output_file}")