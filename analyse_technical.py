"""
Technical Performance Matrix Analysis
=====================================

This script analyzes ONLY the four closed-ended matrix questions:

1. Exercise recognition
2. Corrective feedback accuracy
3. Speech understanding
4. Response speed

It does NOT analyze:
- technical-error open-ended responses
- shared-space comfort
- sports-centre usefulness
- future-use likelihood
- next-study interest
- any other open-ended question

Because these four questions were created specifically for this study,
they are treated as descriptive questionnaire items rather than as a
validated psychometric scale.

Main outputs:
- response counts
- response percentages
- positive / neutral / negative summary
- numeric descriptive summary
- participant-level matrix
- human-readable TXT report

You can manually paste your open-ended qualitative conclusion into the
bottom of the generated TXT report.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PREFERRED_FILE = "Study Data(Post-Study Survey) Technical.csv"

OUTPUT_DIR = Path("analysis_results/technical_matrix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND INPUT FILE
# ============================================================

def resolve_input_file():
    preferred = Path(PREFERRED_FILE)

    if preferred.exists():
        return preferred

    # Fall back to any CSV containing "technical" in the filename.
    matches = [
        p for p in Path(".").glob("*.csv")
        if "technical" in p.name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print("Multiple technical CSV files found:")
        for p in matches:
            print(f" - {p.name}")

        raise FileNotFoundError(
            "\nPlease rename the intended file to:\n"
            f"{PREFERRED_FILE}"
        )

    raise FileNotFoundError(
        "\nCould not find the technical CSV.\n\n"
        "Put this script in the same folder as:\n"
        f"{PREFERRED_FILE}\n"
    )


INPUT_FILE = resolve_input_file()

print(f"Using input file: {INPUT_FILE}")


# ============================================================
# LOAD SURVEYMONKEY CSV
# ============================================================

# Read without headers because SurveyMonkey uses two header rows.
raw = pd.read_csv(INPUT_FILE, header=None)

if raw.shape[1] < 5:
    raise ValueError(
        "The CSV does not contain the expected four matrix questions."
    )

# Row 0 contains the main matrix heading.
# Row 1 contains the four individual statements.
question_row = raw.iloc[1]

# Data starts on row 2.
df = raw.iloc[2:].copy().reset_index(drop=True)

# Only retain:
# column 0 = Participant ID
# columns 1-4 = matrix questions
df = df.iloc[:, 0:5].copy()

QUESTION_NAMES = {
    0: "Participant ID",
    1: "Exercise Recognition",
    2: "Corrective Feedback",
    3: "Speech Understanding",
    4: "Response Speed",
}

QUESTION_TEXT = {
    QUESTION_NAMES[i]: str(question_row.iloc[i]).strip()
    for i in range(1, 5)
}

df.columns = [
    QUESTION_NAMES[i]
    for i in range(5)
]


# ============================================================
# CLEAN PARTICIPANT IDS
# ============================================================

df["Participant ID"] = (
    df["Participant ID"]
    .astype("string")
    .str.strip()
    .str.upper()
    .replace({"": pd.NA})
)

df = df[df["Participant ID"].notna()].copy()

# Save duplicate participant records if any.
duplicates = df[
    df.duplicated("Participant ID", keep=False)
].copy()

if not duplicates.empty:
    duplicates.to_csv(
        OUTPUT_DIR / "technical_matrix_duplicate_participants.csv",
        index=False
    )

# Keep one row per participant.
df = df.drop_duplicates(
    subset="Participant ID",
    keep="first"
).copy()

TOTAL_N = len(df)


# ============================================================
# LIKERT SCALE
# ============================================================

LIKERT_ORDER = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]

LIKERT_SCORE = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

MATRIX_QUESTIONS = [
    "Exercise Recognition",
    "Corrective Feedback",
    "Speech Understanding",
    "Response Speed",
]


# ============================================================
# NORMALIZE RESPONSES
# ============================================================

def normalize_likert(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    lookup = {
        "strongly disagree": "Strongly disagree",
        "disagree": "Disagree",
        "neither agree nor disagree": "Neither agree nor disagree",
        "agree": "Agree",
        "strongly agree": "Strongly agree",
    }

    return lookup.get(text, pd.NA)


for question in MATRIX_QUESTIONS:
    df[question] = df[question].apply(normalize_likert)


# ============================================================
# MISSING-DATA AUDIT
# ============================================================

missing_rows = []

for _, row in df.iterrows():
    for question in MATRIX_QUESTIONS:
        if pd.isna(row[question]):
            missing_rows.append({
                "Participant ID": row["Participant ID"],
                "Question": question,
                "Full Question Text": QUESTION_TEXT[question],
            })

missing_data = pd.DataFrame(
    missing_rows,
    columns=[
        "Participant ID",
        "Question",
        "Full Question Text",
    ]
)

missing_data.to_csv(
    OUTPUT_DIR / "technical_matrix_missing_responses.csv",
    index=False
)


# ============================================================
# PARTICIPANT-LEVEL MATRIX
# ============================================================

participant_matrix = df.copy()

for question in MATRIX_QUESTIONS:
    participant_matrix[f"{question} Score"] = (
        participant_matrix[question]
        .map(LIKERT_SCORE)
    )

participant_matrix = participant_matrix.sort_values(
    "Participant ID"
).reset_index(drop=True)


# ============================================================
# COUNT MATRIX
# ============================================================

count_rows = []

for question in MATRIX_QUESTIONS:
    counts = df[question].value_counts(dropna=False)

    row = {
        "Question": question,
        "Valid N": int(df[question].notna().sum()),
        "Missing N": int(df[question].isna().sum()),
    }

    for response in LIKERT_ORDER:
        row[response] = int(counts.get(response, 0))

    count_rows.append(row)

count_matrix = pd.DataFrame(count_rows)


# ============================================================
# PERCENT MATRIX
# ============================================================

percent_rows = []

for question in MATRIX_QUESTIONS:
    valid_n = int(df[question].notna().sum())

    row = {
        "Question": question,
        "Valid N": valid_n,
    }

    for response in LIKERT_ORDER:
        count = int((df[question] == response).sum())

        row[response] = (
            round(count / valid_n * 100, 1)
            if valid_n > 0
            else np.nan
        )

    percent_rows.append(row)

percent_matrix = pd.DataFrame(percent_rows)


# ============================================================
# POSITIVE / NEUTRAL / NEGATIVE SUMMARY
# ============================================================
#
# Positive = Agree + Strongly agree
# Neutral  = Neither agree nor disagree
# Negative = Disagree + Strongly disagree

summary_rows = []

for question in MATRIX_QUESTIONS:
    valid = df[question].dropna()
    valid_n = len(valid)

    strongly_disagree_n = int((valid == "Strongly disagree").sum())
    disagree_n = int((valid == "Disagree").sum())
    neutral_n = int((valid == "Neither agree nor disagree").sum())
    agree_n = int((valid == "Agree").sum())
    strongly_agree_n = int((valid == "Strongly agree").sum())

    negative_n = strongly_disagree_n + disagree_n
    positive_n = agree_n + strongly_agree_n

    summary_rows.append({
        "Question": question,
        "Valid N": valid_n,

        "Negative n": negative_n,
        "Negative %": (
            round(negative_n / valid_n * 100, 1)
            if valid_n else np.nan
        ),

        "Neutral n": neutral_n,
        "Neutral %": (
            round(neutral_n / valid_n * 100, 1)
            if valid_n else np.nan
        ),

        "Positive n": positive_n,
        "Positive %": (
            round(positive_n / valid_n * 100, 1)
            if valid_n else np.nan
        ),
    })

likert_summary = pd.DataFrame(summary_rows)


# ============================================================
# NUMERIC DESCRIPTIVE SUMMARY
# ============================================================
#
# These are custom Likert items, so the response distributions
# remain the primary result.
#
# Mean/SD/median/IQR are provided as supplementary descriptive
# information only.

numeric_rows = []

for question in MATRIX_QUESTIONS:
    scores = (
        df[question]
        .map(LIKERT_SCORE)
        .dropna()
    )

    if len(scores) == 0:
        continue

    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)

    numeric_rows.append({
        "Question": question,
        "Valid N": len(scores),
        "Mean": round(float(scores.mean()), 3),
        "SD": (
            round(float(scores.std(ddof=1)), 3)
            if len(scores) > 1
            else np.nan
        ),
        "Median": round(float(scores.median()), 3),
        "Q1": round(float(q1), 3),
        "Q3": round(float(q3), 3),
        "IQR": round(float(q3 - q1), 3),
        "Minimum": int(scores.min()),
        "Maximum": int(scores.max()),
    })

numeric_summary = pd.DataFrame(numeric_rows)


# ============================================================
# QUESTION KEY
# ============================================================

question_key = pd.DataFrame([
    {
        "Short Name": question,
        "Full Question": QUESTION_TEXT[question],
    }
    for question in MATRIX_QUESTIONS
])


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

participant_matrix.to_csv(
    OUTPUT_DIR / "technical_matrix_participant_responses.csv",
    index=False
)

count_matrix.to_csv(
    OUTPUT_DIR / "technical_matrix_counts.csv",
    index=False
)

percent_matrix.to_csv(
    OUTPUT_DIR / "technical_matrix_percentages.csv",
    index=False
)

likert_summary.to_csv(
    OUTPUT_DIR / "technical_matrix_positive_negative_summary.csv",
    index=False
)

numeric_summary.to_csv(
    OUTPUT_DIR / "technical_matrix_numeric_summary.csv",
    index=False
)

question_key.to_csv(
    OUTPUT_DIR / "technical_matrix_question_key.csv",
    index=False
)


# ============================================================
# HUMAN-READABLE TXT REPORT
# ============================================================

report = []

report.append("TECHNICAL PERFORMANCE MATRIX ANALYSIS")
report.append("=" * 76)
report.append("")
report.append(f"Participants in dataset: N = {TOTAL_N}")
report.append("")

report.append("ANALYSIS METHOD")
report.append("-" * 76)
report.append(
    "The four custom technical-performance matrix questions were "
    "analyzed descriptively."
)
report.append(
    "For each question, response frequencies (n) and percentages (%) "
    "were calculated across the five Likert response categories."
)
report.append(
    "Positive responses = Agree + Strongly agree."
)
report.append(
    "Neutral responses = Neither agree nor disagree."
)
report.append(
    "Negative responses = Disagree + Strongly disagree."
)
report.append(
    "Because these are custom study questions rather than a validated "
    "multi-item scale, no combined technical-performance score was created."
)
report.append("")

report.append("QUESTIONS")
report.append("-" * 76)

for i, question in enumerate(MATRIX_QUESTIONS, start=1):
    report.append(
        f"{i}. {question}: {QUESTION_TEXT[question]}"
    )

report.append("")
report.append("RESPONSE DISTRIBUTIONS")
report.append("-" * 76)

for question in MATRIX_QUESTIONS:
    report.append("")
    report.append(question)

    row_count = count_matrix[
        count_matrix["Question"] == question
    ].iloc[0]

    row_pct = percent_matrix[
        percent_matrix["Question"] == question
    ].iloc[0]

    report.append(
        f"  Valid N: {int(row_count['Valid N'])}"
    )

    for response in LIKERT_ORDER:
        report.append(
            f"  {response}: "
            f"{int(row_count[response])} "
            f"({row_pct[response]:.1f}%)"
        )

report.append("")
report.append("POSITIVE / NEUTRAL / NEGATIVE SUMMARY")
report.append("-" * 76)

for _, row in likert_summary.iterrows():
    report.append(
        f"{row['Question']}: "
        f"Positive {int(row['Positive n'])}/{int(row['Valid N'])} "
        f"({row['Positive %']:.1f}%), "
        f"Neutral {int(row['Neutral n'])} "
        f"({row['Neutral %']:.1f}%), "
        f"Negative {int(row['Negative n'])} "
        f"({row['Negative %']:.1f}%)"
    )

report.append("")
report.append("NUMERIC DESCRIPTIVE SUMMARY")
report.append("-" * 76)
report.append(
    "Scale: 1=Strongly disagree, 2=Disagree, 3=Neither, "
    "4=Agree, 5=Strongly agree."
)

for _, row in numeric_summary.iterrows():
    report.append(
        f"{row['Question']}: "
        f"M={row['Mean']:.2f}, "
        f"SD={row['SD']:.2f}, "
        f"Median={row['Median']:.2f}, "
        f"IQR={row['IQR']:.2f}"
    )

report.append("")
report.append("MISSING RESPONSES")
report.append("-" * 76)

if missing_data.empty:
    report.append("No matrix responses were missing.")
else:
    report.append(
        f"Missing matrix responses: {len(missing_data)}"
    )

    for _, row in missing_data.iterrows():
        report.append(
            f"  {row['Participant ID']}: {row['Question']}"
        )

report.append("")
report.append("=" * 76)
report.append("OPEN-ENDED RESPONSE ANALYSIS")
report.append("=" * 76)
report.append("")
report.append(
    "PASTE YOUR MANUAL OPEN-ENDED ANALYSIS / CONCLUSION BELOW:"
)
report.append("")
report.append("")
report.append("")
report.append("")
report.append("")

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "technical_matrix_analysis_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print(report_text)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")
