"""
Compare predicted license plate numbers (batch_results.csv) against
ground truth labels (ALL_STATES_MERGED.csv).

Outputs:
  1. Console summary  – overall accuracy + state-wise accuracy table
  2. comparison_results.csv – per-image comparison with match status
  3. state_accuracy.csv     – state-wise accuracy breakdown
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──────────────────────────────────────────────────────────
gt = pd.read_csv(os.path.join(BASE_DIR, "ALL_STATES_MERGED.csv"))
pred = pd.read_csv(os.path.join(BASE_DIR, "batch_results.csv"))

# Normalize column names
gt.columns = gt.columns.str.strip()
pred.columns = pred.columns.str.strip()

# Standardize the image name column
gt.rename(columns={"Image Name": "Image_Name", "Plate Number": "GT_Plate"}, inplace=True)
pred.rename(columns={"Image Name": "Image_Name", "Extracted Plate": "Pred_Plate"}, inplace=True)

# Strip whitespace from values
gt["Image_Name"] = gt["Image_Name"].str.strip()
gt["GT_Plate"] = gt["GT_Plate"].str.strip()
pred["Image_Name"] = pred["Image_Name"].str.strip()
pred["Pred_Plate"] = pred["Pred_Plate"].str.strip()

# ── Merge on Image Name ───────────────────────────────────────────────
merged = pd.merge(gt, pred[["Image_Name", "Pred_Plate", "Evaluation Status"]],
                  on="Image_Name", how="left")

# Mark images that had no prediction
merged["Pred_Plate"] = merged["Pred_Plate"].fillna("NO_PREDICTION")
merged["Evaluation Status"] = merged["Evaluation Status"].fillna("Missing from predictions")

# ── Comparison (case-insensitive) ─────────────────────────────────────
merged["Exact_Match"] = (
    merged["GT_Plate"].str.upper() == merged["Pred_Plate"].str.upper()
)

# ── Overall accuracy ──────────────────────────────────────────────────
total = len(merged)
correct = merged["Exact_Match"].sum()
overall_acc = correct / total * 100

print("=" * 65)
print(f"  OVERALL ACCURACY: {correct}/{total}  =  {overall_acc:.2f}%")
print("=" * 65)

# ── State-wise accuracy ───────────────────────────────────────────────
state_stats = (
    merged.groupby("State")
    .agg(
        Total=("Exact_Match", "count"),
        Correct=("Exact_Match", "sum"),
    )
    .reset_index()
)
state_stats["Accuracy (%)"] = (state_stats["Correct"] / state_stats["Total"] * 100).round(2)
state_stats = state_stats.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

print("\n  STATE-WISE ACCURACY")
print("-" * 50)
print(state_stats.to_string(index=False))
print("-" * 50)

# ── Count stats by evaluation status ──────────────────────────────────
print("\n  EVALUATION STATUS BREAKDOWN")
print("-" * 50)
status_counts = merged["Evaluation Status"].value_counts()
for status, count in status_counts.items():
    print(f"  {status[:60]:60s} : {count}")
print("-" * 50)

# ── Save comparison CSV ───────────────────────────────────────────────
out_comparison = os.path.join(BASE_DIR, "comparison_results.csv")
merged[["State", "Image_Name", "GT_Plate", "Pred_Plate",
        "Exact_Match", "Evaluation Status"]].to_csv(out_comparison, index=False)
print(f"\n✅  Per-image comparison saved → {out_comparison}")

# ── Save state accuracy CSV ───────────────────────────────────────────
out_state = os.path.join(BASE_DIR, "state_accuracy.csv")
state_stats.to_csv(out_state, index=False)
print(f"✅  State-wise accuracy saved  → {out_state}")
