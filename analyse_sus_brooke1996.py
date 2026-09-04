

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Study Data(Post-Study Survey) SUS.csv"

OUTPUT_DIR = Path("analysis_results/sus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RESPONSE SCORING
# ============================================================

LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}


# ============================================================
# LOAD SURVEYMONKEY CSV
# ============================================================

raw = pd.read_csv(INPUT_FILE)

# SurveyMonkey stores the actual 10 SUS item texts in the first
# data row.
item_text_row = raw.iloc[0].copy()

# Remove SurveyMonkey's secondary-header row.
df = raw.iloc[1:].copy().reset_index(drop=True)

participant_col = raw.columns[0]

df[participant_col] = (
    df[participant_col]
    .astype("string")
    .str.strip()
    .str.upper()
    .replace({"": pd.NA})
)

df = df[df[participant_col].notna()].copy()

# Save duplicates if any.
duplicates = df[
    df.duplicated(participant_col, keep=False)
].copy()

if not duplicates.empty:
    duplicates.to_csv(
        OUTPUT_DIR / "sus_duplicate_participants.csv",
        index=False
    )

# Keep one record per participant.
df = df.drop_duplicates(
    subset=participant_col,
    keep="first"
).copy()

TOTAL_N = len(df)


# ============================================================
# IDENTIFY THE 10 SUS ITEMS
# ============================================================

sus_columns = list(raw.columns[1:11])

if len(sus_columns) != 10:
    raise ValueError(
        f"Expected 10 SUS item columns but found {len(sus_columns)}."
    )

item_texts = {}

for number, col in enumerate(sus_columns, start=1):
    text = item_text_row[col]

    if pd.isna(text):
        text = f"SUS Item {number}"

    item_texts[number] = str(text).strip()


# ============================================================
# CREATE CLEAN NUMERIC RESPONSES
# ============================================================

participant_scores = pd.DataFrame()
participant_scores["Participant ID"] = df[participant_col]

missing_audit_rows = []

for item_number, col in enumerate(sus_columns, start=1):

    raw_response = (
        df[col]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA})
    )

    numeric = raw_response.map(LIKERT_MAP)

    # Identify blanks before replacement.
    missing_mask = numeric.isna()

    for participant_id in df.loc[missing_mask, participant_col]:
        missing_audit_rows.append({
            "Participant ID": participant_id,
            "Item Number": item_number,
            "Item Text": item_texts[item_number],
            "Original Response": "Missing",
            "Replacement Response": 3,
            "Replacement Label": "Neither agree nor disagree",
            "Reason":
                "Brooke (1996) instructs use of the centre point "
                "when an item cannot be answered.",
        })

    # Centre-point replacement.
    numeric = numeric.fillna(3).astype(int)

    participant_scores[f"Item {item_number} Raw"] = numeric

    # Original SUS rescoring.
    if item_number % 2 == 1:
        adjusted = numeric - 1
    else:
        adjusted = 5 - numeric

    participant_scores[f"Item {item_number} Adjusted"] = adjusted


# ============================================================
# CALCULATE FINAL SUS SCORE
# ============================================================

adjusted_columns = [
    f"Item {i} Adjusted"
    for i in range(1, 11)
]

participant_scores["Adjusted Sum"] = (
    participant_scores[adjusted_columns]
    .sum(axis=1)
)

participant_scores["SUS Score"] = (
    participant_scores["Adjusted Sum"] * 2.5
)

# Flag participants for whom a centre-point replacement was used.
missing_ids = {
    row["Participant ID"]
    for row in missing_audit_rows
}

participant_scores["Used Centre-Point Replacement"] = (
    participant_scores["Participant ID"]
    .isin(missing_ids)
)

participant_scores = (
    participant_scores
    .sort_values("Participant ID")
    .reset_index(drop=True)
)


# ============================================================
# GROUP-LEVEL DESCRIPTIVE SUMMARY
# ============================================================
#
# Brooke defines the individual SUS score.
# For reporting a study sample, the participant scores are
# summarized descriptively here.

scores = participant_scores["SUS Score"]

q1 = scores.quantile(0.25)
q3 = scores.quantile(0.75)

group_summary = pd.DataFrame(
    [{
        "N": int(scores.count()),
        "Mean SUS": round(float(scores.mean()), 3),
        "SD": round(float(scores.std(ddof=1)), 3)
            if len(scores) > 1 else np.nan,
        "Median SUS": round(float(scores.median()), 3),
        "Q1": round(float(q1), 3),
        "Q3": round(float(q3), 3),
        "IQR": round(float(q3 - q1), 3),
        "Minimum SUS": round(float(scores.min()), 3),
        "Maximum SUS": round(float(scores.max()), 3),
        "Participants with centre-point replacement":
            int(participant_scores[
                "Used Centre-Point Replacement"
            ].sum()),
    }]
)


# ============================================================
# SCORE FREQUENCY TABLE
# ============================================================

score_distribution = (
    participant_scores["SUS Score"]
    .value_counts()
    .sort_index()
    .rename_axis("SUS Score")
    .reset_index(name="n")
)

score_distribution["Percent"] = (
    score_distribution["n"]
    / TOTAL_N
    * 100
).round(1)


# ============================================================
# SCORING KEY / DOCUMENTATION TABLE
# ============================================================

scoring_key_rows = []

for item_number in range(1, 11):

    scoring_key_rows.append({
        "Item Number": item_number,
        "Item Text": item_texts[item_number],
        "Direction":
            "Positive wording"
            if item_number % 2 == 1
            else "Negative wording",
        "Adjusted Score Formula":
            "Raw response - 1"
            if item_number % 2 == 1
            else "5 - Raw response",
    })

scoring_key = pd.DataFrame(scoring_key_rows)


# ============================================================
# MISSING-RESPONSE AUDIT
# ============================================================

missing_audit = pd.DataFrame(
    missing_audit_rows,
    columns=[
        "Participant ID",
        "Item Number",
        "Item Text",
        "Original Response",
        "Replacement Response",
        "Replacement Label",
        "Reason",
    ],
)


# ============================================================
# SAVE RESULTS
# ============================================================

participant_scores.to_csv(
    OUTPUT_DIR / "sus_participant_scores.csv",
    index=False
)

group_summary.to_csv(
    OUTPUT_DIR / "sus_group_summary.csv",
    index=False
)

score_distribution.to_csv(
    OUTPUT_DIR / "sus_score_distribution.csv",
    index=False
)

scoring_key.to_csv(
    OUTPUT_DIR / "sus_scoring_key.csv",
    index=False
)

missing_audit.to_csv(
    OUTPUT_DIR / "sus_missing_response_audit.csv",
    index=False
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

summary = group_summary.iloc[0]

report = []

report.append("SYSTEM USABILITY SCALE (SUS) ANALYSIS")
report.append("=" * 72)
report.append("")

report.append("REFERENCE")
report.append("-" * 72)
report.append(
    'Brooke, J. (1996). SUS: A "Quick and Dirty" Usability Scale.'
)
report.append("")

report.append("SCORING")
report.append("-" * 72)
report.append(
    "Odd items (1,3,5,7,9): adjusted score = response - 1"
)
report.append(
    "Even items (2,4,6,8,10): adjusted score = 5 - response"
)
report.append(
    "Final SUS = sum of the 10 adjusted scores x 2.5"
)
report.append(
    "Possible SUS range = 0 to 100."
)
report.append(
    "The SUS score is a composite scale score, not a percentage."
)
report.append("")

report.append("RESULTS")
report.append("-" * 72)
report.append(f"Participants: N = {int(summary['N'])}")
report.append(
    f"Mean SUS = {summary['Mean SUS']:.2f}"
)
report.append(
    f"SD = {summary['SD']:.2f}"
)
report.append(
    f"Median SUS = {summary['Median SUS']:.2f}"
)
report.append(
    f"IQR = {summary['IQR']:.2f}"
)
report.append(
    f"Range = {summary['Minimum SUS']:.2f} "
    f"to {summary['Maximum SUS']:.2f}"
)
report.append("")

report.append("MISSING RESPONSE HANDLING")
report.append("-" * 72)

replacement_n = int(
    summary["Participants with centre-point replacement"]
)

report.append(
    f"Participants requiring a centre-point replacement: "
    f"{replacement_n}"
)

if missing_audit.empty:
    report.append("No SUS item responses were missing.")
else:
    for _, row in missing_audit.iterrows():
        report.append(
            f"{row['Participant ID']}: "
            f"Item {int(row['Item Number'])} -> "
            f"coded as 3 (centre point)"
        )

report.append("")
report.append("REPORTING NOTE")
report.append("-" * 72)
report.append(
    "Brooke (1996) specifies that SUS should be reported as one "
    "composite usability score. Individual SUS items should not be "
    "interpreted as independent usability outcomes."
)

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "sus_analysis_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report_text)


# ============================================================
# DISPLAY
# ============================================================

print(report_text)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")
