"""
Robot Rep Counter Accuracy Analysis
===================================

This script compares robot rep-count outputs against human video
ground truth.

It is designed for an ONGOING study:
- Missing video rows are allowed.
- New rows can be added later.
- Robot and video data are NOT matched by spreadsheet row position.
- Records are matched by:
      Participant ID + Exercise + Set

PRIMARY ACCURACY METRIC
-----------------------
Count Accuracy (%) = 100 - WAPE

where:

    WAPE = sum(|Robot Count - Video Count|) / sum(Video Count)

Therefore:

    Count Accuracy (%) =
        100 * (1 - sum absolute error / sum video ground-truth reps)

The result is clipped to 0-100%.

This is a COUNTING accuracy measure, not per-repetition classification
accuracy. Per-repetition precision/recall/F1 cannot be calculated from
aggregate set-level counts unless individual reps are temporally aligned.

The script also reports:
- Mean Absolute Error (MAE)
- Exact-match rate
- Within ±1 rep rate
- Signed bias (Robot - Video)
- Per-exercise accuracy

QUALITATIVE NOTES
-----------------
The Notes column is NOT used in the numerical calculations.
A section is left in the TXT report for manually adding explanations
for inaccuracies later.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Study Data(Robot vs Video data).csv"

OUTPUT_DIR = Path("analysis_results/rep_counter_accuracy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND INPUT FILE
# ============================================================

def resolve_input_file():
    preferred = Path(INPUT_FILE)

    if preferred.exists():
        return preferred

    matches = [
        p for p in Path(".").glob("*.csv")
        if "robot" in p.name.lower()
        and "video" in p.name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "\nCould not find the Robot vs Video CSV.\n\n"
        "Place this script in the same folder as:\n"
        f"{INPUT_FILE}\n"
    )


input_path = resolve_input_file()
print(f"Using input file: {input_path}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_path)

required_columns = [
    "Participant",
    "Exercise",
    "Set",
    "Total Attempts",
    "Correct Reps",
    "Incorrect Reps",

    "Participant.1",
    "Exercise.1",
    "Set.1",
    "Total Reps",
    "Correct Reps.1",
    "Incorrect Reps.1",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_participant(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().upper()

    if not text:
        return pd.NA

    return text


EXERCISE_ALIASES = {
    "hummer curl": "hammer curl",
    "hammer curl": "hammer curl",

    "lateral raises": "lateral raise",
    "lateral raise": "lateral raise",

    "squats": "squat",
    "squat": "squat",

    "russian twists": "russian twist",
    "russian twist": "russian twist",

    "leg raise": "leg raises",
    "leg raises": "leg raises",

    "push up": "push-up",
    "push-up": "push-up",
    "pushups": "push-up",
    "push-ups": "push-up",

    "t bar row": "t-bar row",
    "t-bar row": "t-bar row",

    "tricep pushdown": "tricep pushdown",
    "leg extension": "leg extension",
    "chest fly machine": "chest fly machine",
}


def normalize_exercise(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    if not text:
        return pd.NA

    return EXERCISE_ALIASES.get(text, text)


def pretty_exercise(normalized_name):
    pretty = {
        "hammer curl": "Hammer Curl",
        "lateral raise": "Lateral Raise",
        "squat": "Squat",
        "russian twist": "Russian Twist",
        "leg raises": "Leg Raises",
        "push-up": "Push-up",
        "t-bar row": "T-Bar Row",
        "tricep pushdown": "Tricep Pushdown",
        "leg extension": "Leg Extension",
        "chest fly machine": "Chest Fly Machine",
    }

    return pretty.get(
        normalized_name,
        str(normalized_name).title()
    )


# ============================================================
# BUILD ROBOT TABLE
# ============================================================

robot_columns = [
    "Participant",
    "Exercise",
    "Set",
    "Target Reps",
    "Total Attempts",
    "Correct Reps",
    "Incorrect Reps",
]

if "Notes" in df.columns:
    robot_columns.append("Notes")

robot = df[robot_columns].copy()

robot["Participant Key"] = (
    robot["Participant"]
    .apply(normalize_participant)
)

robot["Exercise Key"] = (
    robot["Exercise"]
    .apply(normalize_exercise)
)

robot["Set Key"] = pd.to_numeric(
    robot["Set"],
    errors="coerce"
)

for col in [
    "Target Reps",
    "Total Attempts",
    "Correct Reps",
    "Incorrect Reps",
]:
    if col in robot.columns:
        robot[col] = pd.to_numeric(
            robot[col],
            errors="coerce"
        )


# ============================================================
# BUILD VIDEO GROUND-TRUTH TABLE
# ============================================================

video = df[
    [
        "Participant.1",
        "Exercise.1",
        "Set.1",
        "Total Reps",
        "Correct Reps.1",
        "Incorrect Reps.1",
    ]
].copy()

video.columns = [
    "Video Participant",
    "Video Exercise",
    "Video Set",
    "Video Total Reps",
    "Video Correct Reps",
    "Video Incorrect Reps",
]

video["Participant Key"] = (
    video["Video Participant"]
    .apply(normalize_participant)
)

video["Exercise Key"] = (
    video["Video Exercise"]
    .apply(normalize_exercise)
)

video["Set Key"] = pd.to_numeric(
    video["Video Set"],
    errors="coerce"
)

for col in [
    "Video Total Reps",
    "Video Correct Reps",
    "Video Incorrect Reps",
]:
    video[col] = pd.to_numeric(
        video[col],
        errors="coerce"
    )


# ============================================================
# REMOVE EMPTY / MALFORMED VIDEO RECORDS
# ============================================================

valid_robot_exercises = set(
    robot["Exercise Key"]
    .dropna()
    .unique()
)

video_valid_key_mask = (
    video["Participant Key"].notna()
    & video["Exercise Key"].notna()
    & video["Set Key"].notna()
    & video["Exercise Key"].isin(valid_robot_exercises)
)

malformed_or_empty_video = (
    video.loc[~video_valid_key_mask]
    .copy()
)

video = (
    video.loc[video_valid_key_mask]
    .copy()
)


# ============================================================
# DUPLICATE KEY AUDIT
# ============================================================
#
# A duplicate Participant + Exercise + Set makes matching ambiguous.
# All duplicated keys are EXCLUDED from accuracy until corrected.

key_columns = [
    "Participant Key",
    "Exercise Key",
    "Set Key",
]

robot_duplicate_mask = (
    robot.duplicated(
        subset=key_columns,
        keep=False
    )
)

video_duplicate_mask = (
    video.duplicated(
        subset=key_columns,
        keep=False
    )
)

robot_duplicates = (
    robot.loc[robot_duplicate_mask]
    .copy()
)

video_duplicates = (
    video.loc[video_duplicate_mask]
    .copy()
)

robot_clean = (
    robot.loc[~robot_duplicate_mask]
    .copy()
)

video_clean = (
    video.loc[~video_duplicate_mask]
    .copy()
)


# ============================================================
# MATCH ROBOT TO VIDEO BY PARTICIPANT + EXERCISE + SET
# ============================================================

video_for_merge = video_clean[
    [
        "Participant Key",
        "Exercise Key",
        "Set Key",
        "Video Participant",
        "Video Exercise",
        "Video Set",
        "Video Total Reps",
        "Video Correct Reps",
        "Video Incorrect Reps",
    ]
].copy()

matched = robot_clean.merge(
    video_for_merge,
    on=key_columns,
    how="left",
    indicator=True,
    validate="one_to_one",
)

matched["Exercise"] = (
    matched["Exercise Key"]
    .apply(pretty_exercise)
)

matched_video_mask = (
    matched["_merge"] == "both"
)

unmatched_robot_rows = (
    matched.loc[~matched_video_mask]
    .copy()
)

matched = (
    matched.loc[matched_video_mask]
    .copy()
)


# ============================================================
# ACCURACY FUNCTIONS
# ============================================================

def count_metrics(data, robot_col, video_col):
    """
    Calculate count-level comparison metrics.

    Accuracy = 100 - WAPE
             = 100 * (1 - total absolute error / total ground truth)

    Rows missing either required value are excluded.
    """

    valid = data[
        data[robot_col].notna()
        & data[video_col].notna()
    ].copy()

    if valid.empty:
        return {
            "Valid Sets": 0,
            "Robot Total": np.nan,
            "Video Total": np.nan,
            "Absolute Error Sum": np.nan,
            "MAE": np.nan,
            "Mean Signed Error": np.nan,
            "Count Accuracy %": np.nan,
            "Exact Match n": 0,
            "Exact Match %": np.nan,
            "Within ±1 n": 0,
            "Within ±1 %": np.nan,
        }

    valid["Signed Error"] = (
        valid[robot_col] - valid[video_col]
    )

    valid["Absolute Error"] = (
        valid["Signed Error"].abs()
    )

    n = len(valid)

    robot_total = valid[robot_col].sum()
    video_total = valid[video_col].sum()
    absolute_error_sum = valid["Absolute Error"].sum()

    if video_total > 0:
        accuracy = (
            100
            * (
                1
                - absolute_error_sum / video_total
            )
        )

        accuracy = max(
            0.0,
            min(100.0, accuracy)
        )
    else:
        accuracy = np.nan

    exact_n = int(
        (valid["Absolute Error"] == 0).sum()
    )

    within_one_n = int(
        (valid["Absolute Error"] <= 1).sum()
    )

    return {
        "Valid Sets": n,
        "Robot Total": float(robot_total),
        "Video Total": float(video_total),
        "Absolute Error Sum":
            float(absolute_error_sum),
        "MAE":
            float(valid["Absolute Error"].mean()),
        "Mean Signed Error":
            float(valid["Signed Error"].mean()),
        "Count Accuracy %":
            float(accuracy),
        "Exact Match n": exact_n,
        "Exact Match %":
            exact_n / n * 100,
        "Within ±1 n": within_one_n,
        "Within ±1 %":
            within_one_n / n * 100,
    }


# ============================================================
# OVERALL ANALYSIS
# ============================================================

overall_total = count_metrics(
    matched,
    "Total Attempts",
    "Video Total Reps",
)

overall_correct = count_metrics(
    matched,
    "Correct Reps",
    "Video Correct Reps",
)

# For incorrect reps we avoid calling the percentage result a main
# "accuracy" because many ground-truth incorrect counts are zero.
# MAE/exact agreement are still useful.
overall_incorrect = count_metrics(
    matched,
    "Incorrect Reps",
    "Video Incorrect Reps",
)


overall_summary = pd.DataFrame([
    {
        "Measure": "Total Rep Count",
        **overall_total,
    },
    {
        "Measure": "Correct Rep Count",
        **overall_correct,
    },
    {
        "Measure": "Incorrect Rep Count",
        **overall_incorrect,
    },
])


# ============================================================
# PER-EXERCISE ANALYSIS
# ============================================================

exercise_rows = []

for exercise, group in matched.groupby(
    "Exercise",
    sort=True
):
    total_result = count_metrics(
        group,
        "Total Attempts",
        "Video Total Reps",
    )

    correct_result = count_metrics(
        group,
        "Correct Reps",
        "Video Correct Reps",
    )

    exercise_rows.append({
        "Exercise": exercise,

        "Total Valid Sets":
            total_result["Valid Sets"],

        "Total Count Accuracy %":
            round(
                total_result["Count Accuracy %"],
                2
            )
            if pd.notna(
                total_result["Count Accuracy %"]
            )
            else np.nan,

        "Total Count MAE":
            round(total_result["MAE"], 3)
            if pd.notna(total_result["MAE"])
            else np.nan,

        "Total Exact Match %":
            round(
                total_result["Exact Match %"],
                1
            )
            if pd.notna(
                total_result["Exact Match %"]
            )
            else np.nan,

        "Total Within ±1 %":
            round(
                total_result["Within ±1 %"],
                1
            )
            if pd.notna(
                total_result["Within ±1 %"]
            )
            else np.nan,

        "Correct Count Valid Sets":
            correct_result["Valid Sets"],

        "Correct Rep Count Accuracy %":
            round(
                correct_result["Count Accuracy %"],
                2
            )
            if pd.notna(
                correct_result["Count Accuracy %"]
            )
            else np.nan,

        "Correct Rep MAE":
            round(correct_result["MAE"], 3)
            if pd.notna(correct_result["MAE"])
            else np.nan,

        "Correct Exact Match %":
            round(
                correct_result["Exact Match %"],
                1
            )
            if pd.notna(
                correct_result["Exact Match %"]
            )
            else np.nan,

        "Correct Within ±1 %":
            round(
                correct_result["Within ±1 %"],
                1
            )
            if pd.notna(
                correct_result["Within ±1 %"]
            )
            else np.nan,
    })


per_exercise = pd.DataFrame(
    exercise_rows
)


# ============================================================
# PARTICIPANT / SET-LEVEL ERRORS
# ============================================================

matched["Total Count Error"] = (
    matched["Total Attempts"]
    - matched["Video Total Reps"]
)

matched["Total Count Absolute Error"] = (
    matched["Total Count Error"].abs()
)

matched["Correct Count Error"] = (
    matched["Correct Reps"]
    - matched["Video Correct Reps"]
)

matched["Correct Count Absolute Error"] = (
    matched["Correct Count Error"].abs()
)

matched["Incorrect Count Error"] = (
    matched["Incorrect Reps"]
    - matched["Video Incorrect Reps"]
)

matched["Incorrect Count Absolute Error"] = (
    matched["Incorrect Count Error"].abs()
)


# ============================================================
# SAVE RESULTS
# ============================================================

overall_summary.to_csv(
    OUTPUT_DIR / "rep_accuracy_overall_summary.csv",
    index=False
)

per_exercise.to_csv(
    OUTPUT_DIR / "rep_accuracy_by_exercise.csv",
    index=False
)

matched.to_csv(
    OUTPUT_DIR / "rep_accuracy_matched_sets.csv",
    index=False
)

unmatched_robot_rows.to_csv(
    OUTPUT_DIR / "rep_accuracy_missing_video_rows.csv",
    index=False
)

robot_duplicates.to_csv(
    OUTPUT_DIR / "rep_accuracy_duplicate_robot_keys.csv",
    index=False
)

video_duplicates.to_csv(
    OUTPUT_DIR / "rep_accuracy_duplicate_video_keys.csv",
    index=False
)

malformed_or_empty_video.to_csv(
    OUTPUT_DIR / "rep_accuracy_invalid_or_empty_video_records.csv",
    index=False
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

def fmt(value, decimals=2):
    if pd.isna(value):
        return "NA"

    return f"{value:.{decimals}f}"


report = []

report.append("ROBOT REP COUNTER ACCURACY ANALYSIS")
report.append("=" * 78)
report.append("")

report.append("DATA STATUS")
report.append("-" * 78)
report.append(
    f"Robot records in input: {len(robot)}"
)
report.append(
    f"Usable matched Robot/Video sets: {len(matched)}"
)
report.append(
    f"Robot sets currently without matching video ground truth: "
    f"{len(unmatched_robot_rows)}"
)
report.append(
    f"Robot records excluded because of duplicate keys: "
    f"{len(robot_duplicates)}"
)
report.append(
    f"Video records excluded because of duplicate keys: "
    f"{len(video_duplicates)}"
)
report.append("")

report.append("PRIMARY ACCURACY DEFINITION")
report.append("-" * 78)
report.append(
    "Count Accuracy (%) = "
    "100 × [1 - Σ|Robot Count - Video Count| / ΣVideo Count]"
)
report.append(
    "This is 100 minus weighted absolute percentage error (WAPE)."
)
report.append(
    "Missing video rows are excluded until ground-truth values are added."
)
report.append(
    "Robot and video records are matched using Participant + Exercise + Set."
)
report.append("")

report.append("OVERALL RESULTS")
report.append("-" * 78)

for _, row in overall_summary.iterrows():
    report.append("")
    report.append(row["Measure"])
    report.append(
        f"  Valid sets: {int(row['Valid Sets'])}"
    )
    report.append(
        f"  Count accuracy: "
        f"{fmt(row['Count Accuracy %'])}%"
    )
    report.append(
        f"  Mean absolute error (MAE): "
        f"{fmt(row['MAE'])} reps"
    )
    report.append(
        f"  Mean signed error (Robot - Video): "
        f"{fmt(row['Mean Signed Error'])} reps"
    )
    report.append(
        f"  Exact-match rate: "
        f"{fmt(row['Exact Match %'], 1)}%"
    )
    report.append(
        f"  Within ±1 rep: "
        f"{fmt(row['Within ±1 %'], 1)}%"
    )

report.append("")
report.append("PER-EXERCISE TOTAL REP COUNT ACCURACY")
report.append("-" * 78)

for _, row in per_exercise.iterrows():
    report.append(
        f"{row['Exercise']}: "
        f"N={int(row['Total Valid Sets'])}, "
        f"accuracy={fmt(row['Total Count Accuracy %'])}%, "
        f"MAE={fmt(row['Total Count MAE'])}, "
        f"exact={fmt(row['Total Exact Match %'], 1)}%, "
        f"within ±1={fmt(row['Total Within ±1 %'], 1)}%"
    )

report.append("")
report.append("IMPORTANT INTERPRETATION")
report.append("-" * 78)
report.append(
    "These results measure agreement between aggregate robot counts "
    "and human video counts."
)
report.append(
    "They do not establish per-repetition classification precision, "
    "recall, or F1 because individual robot decisions are not temporally "
    "aligned to individual video repetitions in this dataset."
)
report.append("")

report.append("=" * 78)
report.append("QUALITATIVE NOTES ON INACCURACIES")
report.append("=" * 78)
report.append("")
report.append(
    "PASTE MANUAL QUALITATIVE EXPLANATION BELOW:"
)
report.append("")
report.append(
    "Examples to discuss may include:"
)
report.append(
    "- movements that resembled the intended exercise enough for the "
    "exercise classifier to accept them, but did not satisfy the rep "
    "counter's movement/form thresholds;"
)
report.append(
    "- participants performing an exercise incorrectly or performing a "
    "different movement/exercise;"
)
report.append(
    "- equipment, camera position, range-of-motion, or body-position "
    "constraints that affected counting."
)
report.append("")
report.append("")
report.append("")
report.append("")

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "rep_accuracy_analysis_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)


# ============================================================
# PRINT REPORT
# ============================================================

print()
print(report_text)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")
