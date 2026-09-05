"""
Rep Counter Accuracy Analysis
=============================

Designed for an ongoing HRI study comparing robot output with
human-coded video ground truth.

IMPORTANT DESIGN DECISIONS
--------------------------
1. Robot and video records are NOT compared by spreadsheet row.
2. They are matched using:
       Participant ID + Exercise + Set
3. Missing video ground truth is allowed and is excluded only from
   the metric that requires it.
4. Duplicate keys are flagged and excluded rather than guessed.
5. The PRIMARY metric is Correct Rep Count Accuracy because the
   system's meaningful output is the number of repetitions accepted
   as correctly performed.
6. Total Attempt Count Accuracy is reported as a SECONDARY metric.
7. The script does NOT dynamically choose whichever metric gives the
   largest percentage. Both are always reported.

COUNT ACCURACY
--------------
Count Accuracy (%) =
    100 * [1 - sum(|Robot - Video|) / sum(Video)]

This is 100 - WAPE (Weighted Absolute Percentage Error).

Additional agreement measures:
- Mean Absolute Error (MAE)
- Mean signed error / bias (Robot - Video)
- Exact-match rate
- Within ±1 rep rate

This is set-level COUNT agreement. Precision, recall, and F1 require
individual repetition-level temporal labels and cannot be correctly
derived from aggregate set counts alone.
"""

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PREFERRED_INPUT_FILE = "Study Data(Robot vs Video data)(1).csv"

OUTPUT_DIR = Path("analysis_results/rep_counter_accuracy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILE
# ============================================================

def resolve_input_file():
    preferred = Path(PREFERRED_INPUT_FILE)

    if preferred.exists():
        return preferred

    candidates = sorted(
        [
            p for p in Path(".").glob("*.csv")
            if "robot" in p.name.lower()
            and "video" in p.name.lower()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        print("Multiple Robot-vs-Video CSV files were found:")
        for p in candidates:
            print(f" - {p.name}")

        raise FileNotFoundError(
            "\nRename the cleaned/current file to:\n"
            f"{PREFERRED_INPUT_FILE}"
        )

    raise FileNotFoundError(
        "\nCould not find the Robot-vs-Video CSV.\n"
        f"Expected: {PREFERRED_INPUT_FILE}\n"
    )


INPUT_FILE = resolve_input_file()
print(f"Using input file: {INPUT_FILE}")


# ============================================================
# NORMALIZATION
# ============================================================

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


PRETTY_EXERCISE = {
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


def normalize_participant(value):
    value = (value or "").strip().upper()
    return value or None


def normalize_exercise(value):
    value = (value or "").strip().lower()

    if not value:
        return None

    return EXERCISE_ALIASES.get(value, value)


def display_exercise(value):
    return PRETTY_EXERCISE.get(value, str(value).title())


def to_number(value):
    text = str(value or "").strip()

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def normalize_set(value):
    number = to_number(value)

    if number is None:
        return None

    if float(number).is_integer():
        return int(number)

    return number


def make_key(participant, exercise, set_number):
    p = normalize_participant(participant)
    e = normalize_exercise(exercise)
    s = normalize_set(set_number)

    if p is None or e is None or s is None:
        return None

    return (p, e, s)


# ============================================================
# READ CSV BY POSITION
# ============================================================
#
# We intentionally do NOT use column-name suffixes such as Participant.1.
# The exported CSV contains duplicate header names on the Robot and Video
# sides. Reading by position is much more stable.
#
# Expected layout:
#   0 Participant       Robot
#   1 Exercise          Robot
#   2 Set               Robot
#   3 Target Reps       Robot
#   4 Total Attempts    Robot
#   5 Correct Reps      Robot
#   6 Incorrect Reps    Robot
#
#   7 Participant       Video
#   8 Exercise          Video
#   9 Set               Video
#  10 Total Reps        Video
#  11 Correct Reps      Video
#  12 Incorrect Reps    Video
#
#  13 Notes

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

if not rows:
    raise ValueError("The input CSV is empty.")

header = rows[0]
data_rows = rows[1:]

if len(header) < 13:
    raise ValueError(
        "Expected at least 13 columns in the Robot-vs-Video CSV."
    )


robot_records = []
video_records = []

for csv_row_number, row in enumerate(data_rows, start=2):

    if len(row) < 14:
        row = row + [""] * (14 - len(row))

    robot_key = make_key(
        row[0],
        row[1],
        row[2],
    )

    if robot_key is not None:
        robot_records.append({
            "Key": robot_key,
            "CSV Row": csv_row_number,
            "Participant": normalize_participant(row[0]),
            "Exercise": normalize_exercise(row[1]),
            "Set": normalize_set(row[2]),
            "Target Reps": to_number(row[3]),
            "Total Attempts": to_number(row[4]),
            "Correct Reps": to_number(row[5]),
            "Incorrect Reps": to_number(row[6]),
            "Notes": row[13].strip(),
        })

    video_key = make_key(
        row[7],
        row[8],
        row[9],
    )

    if video_key is not None:
        video_records.append({
            "Key": video_key,
            "CSV Row": csv_row_number,
            "Participant": normalize_participant(row[7]),
            "Exercise": normalize_exercise(row[8]),
            "Set": normalize_set(row[9]),
            "Total Reps": to_number(row[10]),
            "Correct Reps": to_number(row[11]),
            "Incorrect Reps": to_number(row[12]),
        })


# ============================================================
# DUPLICATE KEY AUDIT
# ============================================================

robot_key_counts = Counter(
    record["Key"]
    for record in robot_records
)

video_key_counts = Counter(
    record["Key"]
    for record in video_records
)


robot_duplicates = [
    record for record in robot_records
    if robot_key_counts[record["Key"]] > 1
]

video_duplicates = [
    record for record in video_records
    if video_key_counts[record["Key"]] > 1
]


# Only unambiguous records enter the quantitative analysis.
robot_map = {
    record["Key"]: record
    for record in robot_records
    if robot_key_counts[record["Key"]] == 1
}

video_map = {
    record["Key"]: record
    for record in video_records
    if video_key_counts[record["Key"]] == 1
}


robot_keys = set(robot_map)
video_keys = set(video_map)

matched_keys = sorted(
    robot_keys & video_keys
)

robot_only_keys = sorted(
    robot_keys - video_keys
)

video_only_keys = sorted(
    video_keys - robot_keys
)


# ============================================================
# BUILD MATCHED SET TABLE
# ============================================================

matched_sets = []

for key in matched_keys:
    robot = robot_map[key]
    video = video_map[key]

    matched_sets.append({
        "Participant": key[0],
        "Exercise": display_exercise(key[1]),
        "Exercise Key": key[1],
        "Set": key[2],

        "Robot CSV Row": robot["CSV Row"],
        "Video CSV Row": video["CSV Row"],

        "Target Reps": robot["Target Reps"],

        "Robot Total Attempts": robot["Total Attempts"],
        "Video Total Reps": video["Total Reps"],

        "Robot Correct Reps": robot["Correct Reps"],
        "Video Correct Reps": video["Correct Reps"],

        "Robot Incorrect Reps": robot["Incorrect Reps"],
        "Video Incorrect Reps": video["Incorrect Reps"],

        "Notes": robot["Notes"],
    })


# ============================================================
# COUNT METRICS
# ============================================================

def calculate_metrics(records, robot_field, video_field):
    valid = []

    for row in records:
        robot_value = row[robot_field]
        video_value = row[video_field]

        if robot_value is None or video_value is None:
            continue

        error = robot_value - video_value

        valid.append({
            "Robot": robot_value,
            "Video": video_value,
            "Signed Error": error,
            "Absolute Error": abs(error),
        })

    if not valid:
        return {
            "Valid Sets": 0,
            "Robot Total": None,
            "Video Total": None,
            "Absolute Error Sum": None,
            "WAPE %": None,
            "Count Accuracy %": None,
            "MAE": None,
            "Mean Signed Error": None,
            "Exact Match n": 0,
            "Exact Match %": None,
            "Within ±1 n": 0,
            "Within ±1 %": None,
        }

    n = len(valid)

    robot_total = sum(
        row["Robot"]
        for row in valid
    )

    video_total = sum(
        row["Video"]
        for row in valid
    )

    absolute_error_sum = sum(
        row["Absolute Error"]
        for row in valid
    )

    if video_total > 0:
        wape = (
            absolute_error_sum
            / video_total
            * 100
        )

        accuracy = 100 - wape

        # A count accuracy below 0% has no useful interpretation.
        accuracy = max(
            0.0,
            min(100.0, accuracy)
        )
    else:
        wape = None
        accuracy = None

    mae = statistics.mean(
        row["Absolute Error"]
        for row in valid
    )

    bias = statistics.mean(
        row["Signed Error"]
        for row in valid
    )

    exact_n = sum(
        row["Absolute Error"] == 0
        for row in valid
    )

    within_one_n = sum(
        row["Absolute Error"] <= 1
        for row in valid
    )

    return {
        "Valid Sets": n,
        "Robot Total": robot_total,
        "Video Total": video_total,
        "Absolute Error Sum": absolute_error_sum,
        "WAPE %": wape,
        "Count Accuracy %": accuracy,
        "MAE": mae,
        "Mean Signed Error": bias,
        "Exact Match n": exact_n,
        "Exact Match %": exact_n / n * 100,
        "Within ±1 n": within_one_n,
        "Within ±1 %": within_one_n / n * 100,
    }


# ============================================================
# OVERALL RESULTS
# ============================================================

correct_metrics = calculate_metrics(
    matched_sets,
    "Robot Correct Reps",
    "Video Correct Reps",
)

total_metrics = calculate_metrics(
    matched_sets,
    "Robot Total Attempts",
    "Video Total Reps",
)

incorrect_metrics = calculate_metrics(
    matched_sets,
    "Robot Incorrect Reps",
    "Video Incorrect Reps",
)


overall_results = [
    {
        "Role": "PRIMARY",
        "Measure": "Correct Rep Count",
        **correct_metrics,
    },
    {
        "Role": "SECONDARY",
        "Measure": "Total Attempt Count",
        **total_metrics,
    },
    {
        "Role": "DIAGNOSTIC",
        "Measure": "Incorrect Rep Count",
        **incorrect_metrics,
    },
]


# ============================================================
# ADD SET-LEVEL ERRORS
# ============================================================

for row in matched_sets:

    if (
        row["Robot Correct Reps"] is not None
        and row["Video Correct Reps"] is not None
    ):
        row["Correct Rep Error"] = (
            row["Robot Correct Reps"]
            - row["Video Correct Reps"]
        )

        row["Correct Rep Absolute Error"] = abs(
            row["Correct Rep Error"]
        )
    else:
        row["Correct Rep Error"] = None
        row["Correct Rep Absolute Error"] = None

    if (
        row["Robot Total Attempts"] is not None
        and row["Video Total Reps"] is not None
    ):
        row["Total Attempt Error"] = (
            row["Robot Total Attempts"]
            - row["Video Total Reps"]
        )

        row["Total Attempt Absolute Error"] = abs(
            row["Total Attempt Error"]
        )
    else:
        row["Total Attempt Error"] = None
        row["Total Attempt Absolute Error"] = None


# ============================================================
# PER-EXERCISE RESULTS
# ============================================================

exercise_groups = defaultdict(list)

for row in matched_sets:
    exercise_groups[row["Exercise"]].append(row)


per_exercise_results = []

for exercise in sorted(exercise_groups):
    group = exercise_groups[exercise]

    correct = calculate_metrics(
        group,
        "Robot Correct Reps",
        "Video Correct Reps",
    )

    total = calculate_metrics(
        group,
        "Robot Total Attempts",
        "Video Total Reps",
    )

    per_exercise_results.append({
        "Exercise": exercise,

        "Correct Valid Sets":
            correct["Valid Sets"],

        "Correct Rep Accuracy %":
            correct["Count Accuracy %"],

        "Correct Rep MAE":
            correct["MAE"],

        "Correct Exact Match %":
            correct["Exact Match %"],

        "Correct Within ±1 %":
            correct["Within ±1 %"],

        "Total Valid Sets":
            total["Valid Sets"],

        "Total Attempt Accuracy %":
            total["Count Accuracy %"],

        "Total Attempt MAE":
            total["MAE"],

        "Total Exact Match %":
            total["Exact Match %"],

        "Total Within ±1 %":
            total["Within ±1 %"],
    })


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns = list(rows[0].keys())

    with open(
        path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

write_csv(
    OUTPUT_DIR / "rep_accuracy_overall_summary.csv",
    overall_results,
)

write_csv(
    OUTPUT_DIR / "rep_accuracy_by_exercise.csv",
    per_exercise_results,
)

write_csv(
    OUTPUT_DIR / "rep_accuracy_matched_sets.csv",
    matched_sets,
)

write_csv(
    OUTPUT_DIR / "rep_accuracy_duplicate_robot_keys.csv",
    robot_duplicates,
)

write_csv(
    OUTPUT_DIR / "rep_accuracy_duplicate_video_keys.csv",
    video_duplicates,
)


unmatched_rows = []

for key in robot_only_keys:
    record = robot_map[key]

    unmatched_rows.append({
        "Side": "Robot only",
        "Participant": key[0],
        "Exercise": display_exercise(key[1]),
        "Set": key[2],
        "CSV Row": record["CSV Row"],
    })

for key in video_only_keys:
    record = video_map[key]

    unmatched_rows.append({
        "Side": "Video only",
        "Participant": key[0],
        "Exercise": display_exercise(key[1]),
        "Set": key[2],
        "CSV Row": record["CSV Row"],
    })


write_csv(
    OUTPUT_DIR / "rep_accuracy_unmatched_keys.csv",
    unmatched_rows,
)


# ============================================================
# REPORT
# ============================================================

def fmt(value, decimals=2):
    if value is None:
        return "NA"

    try:
        if math.isnan(value):
            return "NA"
    except TypeError:
        pass

    return f"{value:.{decimals}f}"


report = []

report.append("ROBOT REP COUNTER ACCURACY ANALYSIS")
report.append("=" * 80)
report.append("")

report.append("DATA MATCHING")
report.append("-" * 80)
report.append(
    f"Robot records with valid keys: {len(robot_records)}"
)
report.append(
    f"Video records with valid keys: {len(video_records)}"
)
report.append(
    f"Matched unambiguous sets: {len(matched_sets)}"
)
report.append(
    f"Robot-only unmatched keys: {len(robot_only_keys)}"
)
report.append(
    f"Video-only unmatched keys: {len(video_only_keys)}"
)
report.append(
    f"Robot records involved in duplicate keys: {len(robot_duplicates)}"
)
report.append(
    f"Video records involved in duplicate keys: {len(video_duplicates)}"
)
report.append("")

report.append("PRIMARY RESULT: CORRECT REP COUNT ACCURACY")
report.append("-" * 80)
report.append(
    "This is the primary metric because the system's meaningful "
    "output is whether a movement is accepted as a correctly "
    "performed repetition."
)
report.append(
    f"Valid sets: {correct_metrics['Valid Sets']}"
)
report.append(
    f"Correct rep count accuracy: "
    f"{fmt(correct_metrics['Count Accuracy %'])}%"
)
report.append(
    f"MAE: {fmt(correct_metrics['MAE'])} reps/set"
)
report.append(
    f"Mean signed error (Robot - Video): "
    f"{fmt(correct_metrics['Mean Signed Error'])} reps/set"
)
report.append(
    f"Exact-match rate: "
    f"{fmt(correct_metrics['Exact Match %'], 1)}%"
)
report.append(
    f"Within ±1 rep: "
    f"{fmt(correct_metrics['Within ±1 %'], 1)}%"
)
report.append("")

report.append("SECONDARY RESULT: TOTAL ATTEMPT COUNT")
report.append("-" * 80)
report.append(
    "Total attempts include movements that may resemble the exercise "
    "but can legitimately be rejected by the form/rep logic."
)
report.append(
    f"Valid sets: {total_metrics['Valid Sets']}"
)
report.append(
    f"Total attempt count accuracy: "
    f"{fmt(total_metrics['Count Accuracy %'])}%"
)
report.append(
    f"MAE: {fmt(total_metrics['MAE'])} reps/set"
)
report.append(
    f"Mean signed error (Robot - Video): "
    f"{fmt(total_metrics['Mean Signed Error'])} reps/set"
)
report.append(
    f"Exact-match rate: "
    f"{fmt(total_metrics['Exact Match %'], 1)}%"
)
report.append(
    f"Within ±1 rep: "
    f"{fmt(total_metrics['Within ±1 %'], 1)}%"
)
report.append("")

report.append("ACCURACY FORMULA")
report.append("-" * 80)
report.append(
    "Count Accuracy (%) = "
    "100 × [1 - Σ|Robot Count - Video Count| / ΣVideo Count]"
)
report.append(
    "This is 100 minus weighted absolute percentage error (WAPE)."
)
report.append("")

report.append("PER-EXERCISE CORRECT REP ACCURACY")
report.append("-" * 80)

for row in per_exercise_results:
    report.append(
        f"{row['Exercise']}: "
        f"N={row['Correct Valid Sets']}, "
        f"accuracy={fmt(row['Correct Rep Accuracy %'])}%, "
        f"MAE={fmt(row['Correct Rep MAE'])}, "
        f"exact={fmt(row['Correct Exact Match %'], 1)}%, "
        f"within ±1={fmt(row['Correct Within ±1 %'], 1)}%"
    )

report.append("")

if unmatched_rows or robot_duplicates or video_duplicates:
    report.append("DATA QUALITY WARNING")
    report.append("-" * 80)
    report.append(
        "Some keys are still unmatched or duplicated. These records "
        "are excluded rather than guessed. See the audit CSV files."
    )
    report.append("")

report.append("=" * 80)
report.append("QUALITATIVE EXPLANATION OF INACCURACIES")
report.append("=" * 80)
report.append("")
report.append(
    "PASTE / EDIT YOUR MANUAL VIDEO-REVIEW CONCLUSION BELOW:"
)
report.append("")
report.append(
    "- Some movements resembled the intended exercise enough for the "
    "exercise classifier to accept the exercise, but did not satisfy "
    "the repetition/form thresholds and were therefore rejected."
)
report.append(
    "- Some participants performed the movement incorrectly or "
    "performed a substantially different movement."
)
report.append(
    "- Equipment constraints, range of motion, camera position, and "
    "body position may explain some additional count discrepancies."
)
report.append("")
report.append("")
report.append("")

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "rep_accuracy_analysis_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(report_text)


# ============================================================
# PRINT
# ============================================================

print()
print(report_text)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")
