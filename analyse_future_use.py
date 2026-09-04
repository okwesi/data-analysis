"""
Future Use Analysis
===================

Analyzes ONLY the closed-ended future-use questions:

1. Could the robot assist people in the sports centre?
2. How likely would the participant be to use the robot in future
   exercise sessions if it were available at no extra cost?
3. Would the participant be interested in a future improved study?

The open-ended "explain why" responses are NOT automatically analyzed.
A blank section is added to the TXT report so a manually written
qualitative conclusion can be pasted there later.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Study Data(Post-Study Survey) Future Use(1).csv"

OUTPUT_DIR = Path("analysis_results/future_use")
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
        if "future" in p.name.lower()
        and "use" in p.name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "\nCould not automatically find the Future Use CSV.\n\n"
        "Put this script in the same folder as:\n"
        f"{INPUT_FILE}\n"
    )


input_path = resolve_input_file()

print(f"Using input file: {input_path}")


# ============================================================
# LOAD SURVEYMONKEY CSV
# ============================================================

raw = pd.read_csv(input_path, header=None)

if raw.shape[1] < 5:
    raise ValueError(
        "Expected at least 5 columns in the Future Use CSV."
    )

# SurveyMonkey:
# row 0 = question text
# row 1 = response type
# row 2 onward = participant data

question_row = raw.iloc[0].copy()

df = raw.iloc[2:, 0:5].copy().reset_index(drop=True)

df.columns = [
    "Participant ID",
    "Assist Sports Centre",
    "Open-Ended Explanation",
    "Future Use Likelihood",
    "Interested in Next Study",
]


# ============================================================
# CLEAN DATA
# ============================================================

def clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA})
    )


df["Participant ID"] = (
    clean_text(df["Participant ID"])
    .str.upper()
)

df = df[df["Participant ID"].notna()].copy()

for col in [
    "Assist Sports Centre",
    "Open-Ended Explanation",
    "Future Use Likelihood",
    "Interested in Next Study",
]:
    df[col] = clean_text(df[col])

# Save duplicate records if present.
duplicates = df[
    df.duplicated("Participant ID", keep=False)
].copy()

if not duplicates.empty:
    duplicates.to_csv(
        OUTPUT_DIR / "future_use_duplicate_participants.csv",
        index=False
    )

df = df.drop_duplicates(
    subset="Participant ID",
    keep="first"
).copy()

TOTAL_N = len(df)


# ============================================================
# QUESTION TEXT
# ============================================================

QUESTION_TEXT = {
    "Assist Sports Centre": str(question_row.iloc[1]).strip(),
    "Future Use Likelihood": str(question_row.iloc[3]).strip(),
    "Interested in Next Study": str(question_row.iloc[4]).strip(),
}


# ============================================================
# GENERIC FREQUENCY TABLE
# ============================================================

def frequency_table(series, question_name, category_order=None):
    valid = series.dropna()
    valid_n = len(valid)
    missing_n = TOTAL_N - valid_n

    if category_order is None:
        categories = list(valid.value_counts().index)
    else:
        categories = category_order

    rows = []

    for category in categories:
        count = int((valid == category).sum())

        rows.append({
            "Question": question_name,
            "Response": category,
            "n": count,
            "Percent of valid responses": (
                round(count / valid_n * 100, 1)
                if valid_n
                else np.nan
            ),
            "Percent of total sample": (
                round(count / TOTAL_N * 100, 1)
                if TOTAL_N
                else np.nan
            ),
            "Valid N": valid_n,
            "Missing N": missing_n,
            "Total N": TOTAL_N,
        })

    if missing_n > 0:
        rows.append({
            "Question": question_name,
            "Response": "Missing",
            "n": missing_n,
            "Percent of valid responses": np.nan,
            "Percent of total sample": round(
                missing_n / TOTAL_N * 100,
                1
            ),
            "Valid N": valid_n,
            "Missing N": missing_n,
            "Total N": TOTAL_N,
        })

    return pd.DataFrame(rows)


# ============================================================
# ASSIST SPORTS CENTRE
# ============================================================

assist_order = [
    "Yes",
    "No",
    "Not sure",
]

assist_summary = frequency_table(
    df["Assist Sports Centre"],
    "Could assist people in the sports centre",
    assist_order,
)


# ============================================================
# FUTURE USE LIKELIHOOD
# ============================================================

likelihood_order = [
    "Very unlikely",
    "Unlikely",
    "Neither likely nor unlikely",
    "Likely",
    "Very likely",
]

likelihood_summary = frequency_table(
    df["Future Use Likelihood"],
    "Likelihood of future use",
    likelihood_order,
)


# ============================================================
# COLLAPSED FUTURE-USE LIKELIHOOD
# ============================================================
#
# Positive = Likely + Very likely
# Neutral  = Neither likely nor unlikely
# Negative = Unlikely + Very unlikely

valid_likelihood = df["Future Use Likelihood"].dropna()
valid_likelihood_n = len(valid_likelihood)

positive_n = int(
    valid_likelihood.isin(
        ["Likely", "Very likely"]
    ).sum()
)

neutral_n = int(
    (
        valid_likelihood
        == "Neither likely nor unlikely"
    ).sum()
)

negative_n = int(
    valid_likelihood.isin(
        ["Unlikely", "Very unlikely"]
    ).sum()
)

likelihood_collapsed = pd.DataFrame([
    {
        "Category": "Positive (Likely + Very likely)",
        "n": positive_n,
        "Percent": round(
            positive_n / valid_likelihood_n * 100,
            1
        ) if valid_likelihood_n else np.nan,
        "Valid N": valid_likelihood_n,
    },
    {
        "Category": "Neutral",
        "n": neutral_n,
        "Percent": round(
            neutral_n / valid_likelihood_n * 100,
            1
        ) if valid_likelihood_n else np.nan,
        "Valid N": valid_likelihood_n,
    },
    {
        "Category": "Negative (Unlikely + Very unlikely)",
        "n": negative_n,
        "Percent": round(
            negative_n / valid_likelihood_n * 100,
            1
        ) if valid_likelihood_n else np.nan,
        "Valid N": valid_likelihood_n,
    },
])


# ============================================================
# NEXT STUDY INTEREST
# ============================================================

next_study_order = [
    "Yes",
    "No",
]

next_study_summary = frequency_table(
    df["Interested in Next Study"],
    "Interested in next improved study",
    next_study_order,
)


# ============================================================
# COMBINED CLOSED-ENDED SUMMARY
# ============================================================

closed_ended_summary = pd.concat(
    [
        assist_summary,
        likelihood_summary,
        next_study_summary,
    ],
    ignore_index=True,
)


# ============================================================
# PARTICIPANT-LEVEL CLOSED-ENDED DATA
# ============================================================

participant_data = df[
    [
        "Participant ID",
        "Assist Sports Centre",
        "Future Use Likelihood",
        "Interested in Next Study",
    ]
].sort_values(
    "Participant ID"
).reset_index(drop=True)


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

closed_ended_summary.to_csv(
    OUTPUT_DIR / "future_use_closed_ended_summary.csv",
    index=False
)

assist_summary.to_csv(
    OUTPUT_DIR / "future_use_assist_percentages.csv",
    index=False
)

likelihood_summary.to_csv(
    OUTPUT_DIR / "future_use_likelihood_percentages.csv",
    index=False
)

likelihood_collapsed.to_csv(
    OUTPUT_DIR / "future_use_likelihood_collapsed.csv",
    index=False
)

next_study_summary.to_csv(
    OUTPUT_DIR / "future_use_next_study_percentages.csv",
    index=False
)

participant_data.to_csv(
    OUTPUT_DIR / "future_use_participant_responses.csv",
    index=False
)


# ============================================================
# HUMAN-READABLE TXT REPORT
# ============================================================

assist_yes = int(
    (df["Assist Sports Centre"] == "Yes").sum()
)

assist_no = int(
    (df["Assist Sports Centre"] == "No").sum()
)

assist_unsure = int(
    (df["Assist Sports Centre"] == "Not sure").sum()
)

next_yes = int(
    (df["Interested in Next Study"] == "Yes").sum()
)

next_no = int(
    (df["Interested in Next Study"] == "No").sum()
)

future_missing = int(
    df["Future Use Likelihood"].isna().sum()
)

report = []

report.append("FUTURE USE ANALYSIS")
report.append("=" * 76)
report.append("")
report.append(f"Participants: N = {TOTAL_N}")
report.append("")

report.append("1. COULD THE ROBOT ASSIST PEOPLE IN THE SPORTS CENTRE?")
report.append("-" * 76)
report.append(
    f"Yes: {assist_yes}/{TOTAL_N} "
    f"({assist_yes / TOTAL_N * 100:.1f}%)"
)
report.append(
    f"No: {assist_no}/{TOTAL_N} "
    f"({assist_no / TOTAL_N * 100:.1f}%)"
)
report.append(
    f"Not sure: {assist_unsure}/{TOTAL_N} "
    f"({assist_unsure / TOTAL_N * 100:.1f}%)"
)
report.append("")

report.append("2. LIKELIHOOD OF FUTURE USE")
report.append("-" * 76)

for _, row in likelihood_summary.iterrows():
    if row["Response"] == "Missing":
        continue

    report.append(
        f"{row['Response']}: "
        f"{int(row['n'])}/{int(row['Valid N'])} "
        f"({row['Percent of valid responses']:.1f}%)"
    )

if future_missing:
    report.append(
        f"Missing response: {future_missing}/{TOTAL_N} "
        f"({future_missing / TOTAL_N * 100:.1f}%)"
    )

report.append("")
report.append("Collapsed likelihood:")
report.append(
    f"Positive (Likely + Very likely): "
    f"{positive_n}/{valid_likelihood_n} "
    f"({positive_n / valid_likelihood_n * 100:.1f}%)"
)
report.append(
    f"Neutral: "
    f"{neutral_n}/{valid_likelihood_n} "
    f"({neutral_n / valid_likelihood_n * 100:.1f}%)"
)
report.append(
    f"Negative (Unlikely + Very unlikely): "
    f"{negative_n}/{valid_likelihood_n} "
    f"({negative_n / valid_likelihood_n * 100:.1f}%)"
)
report.append("")

report.append("3. INTEREST IN A FUTURE IMPROVED STUDY")
report.append("-" * 76)
report.append(
    f"Yes: {next_yes}/{TOTAL_N} "
    f"({next_yes / TOTAL_N * 100:.1f}%)"
)
report.append(
    f"No: {next_no}/{TOTAL_N} "
    f"({next_no / TOTAL_N * 100:.1f}%)"
)
report.append("")

report.append("=" * 76)
report.append("OPEN-ENDED FUTURE-USE CONCLUSION")
report.append("=" * 76)
report.append("")
report.append(
    "PASTE THE MANUAL QUALITATIVE CONCLUSION BELOW:"
)
report.append("")
report.append("")
report.append("")
report.append("")
report.append("")

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "future_use_analysis_report.txt",
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
