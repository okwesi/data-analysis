import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Study Data(Demographics).csv"

OUTPUT_DIR = Path("analysis_results/demographics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD SURVEYMONKEY CSV
# ============================================================

# SurveyMonkey export structure:
# - Row 1 contains the main question headings
# - Row 2 contains response labels / checkbox option labels
#
# Pandas uses Row 1 as the header, so the first dataframe row
# is the SurveyMonkey secondary header. We remove it below.

raw = pd.read_csv(INPUT_FILE)

# Save the option labels before removing the SurveyMonkey sub-header.
subheader = raw.iloc[0].copy()

# Remove SurveyMonkey's secondary header row.
df = raw.iloc[1:].copy().reset_index(drop=True)

# ============================================================
# CLEAN PARTICIPANT IDS
# ============================================================

participant_col = "Participant ID"

df[participant_col] = (
    df[participant_col]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Remove blank/invalid participant IDs.
df = df[
    df[participant_col].notna()
    & (df[participant_col] != "")
    & (df[participant_col] != "NAN")
].copy()

# Report duplicate IDs before removing them.
duplicate_rows = df[df.duplicated(subset=[participant_col], keep=False)].copy()

if not duplicate_rows.empty:
    duplicate_rows.to_csv(
        OUTPUT_DIR / "duplicate_participant_records.csv",
        index=False
    )

# Keep one record per participant.
# If duplicate rows are identical, this simply removes the duplicate.
df = df.drop_duplicates(subset=[participant_col], keep="first").copy()

N = len(df)

# ============================================================
# COLUMN DEFINITIONS FROM THIS SURVEY EXPORT
# ============================================================

AGE_COL = "Please provide your age range?"
GENDER_COL = "please provideyour gender?"
GENDER_OTHER_COL = "Unnamed: 4"

STUDENT_COL = "Are you currently a student?"
OCCUPATION_COL = "What is your current occupation?"

EXERCISE_FREQUENCY_COL = (
    "How often do you typically engage in physical exercise "
    "or physical activity?"
)
EXERCISE_FREQUENCY_OTHER_COL = "Unnamed: 8"

# SurveyMonkey exported the multi-select activity question
# across columns 9-20. The first row of the CSV contains the
# actual option names.
ACTIVITY_COLS = [
    "What types of physical physical activity do you typically engage in?",
    "Unnamed: 10",
    "Unnamed: 11",
    "Unnamed: 12",
    "Unnamed: 13",
    "Unnamed: 14",
    "Unnamed: 15",
    "Unnamed: 16",
    "Unnamed: 17",
    "Unnamed: 18",
    "Unnamed: 19",
]

ACTIVITY_OTHER_COL = "Unnamed: 20"

# These fields are personally identifying and are intentionally
# excluded from analysis outputs.
NAME_COL = "Please provide your full name (first name, last name)"
EMAIL_COL = "At what email address would you like to be contacted?"

# ============================================================
# HELPERS
# ============================================================

def clean_text(series):
    """Strip whitespace and convert empty strings to missing."""
    s = series.astype("string").str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})


def frequency_table(series, variable_name, denominator="all"):
    """
    Produce counts and percentages.

    denominator='all':
        percentage is based on all unique participants (N)

    denominator='valid':
        percentage is based only on non-missing responses
    """
    s = clean_text(series)
    counts = s.value_counts(dropna=True)

    valid_n = int(s.notna().sum())
    missing_n = int(s.isna().sum())

    denom = N if denominator == "all" else valid_n

    rows = []
    for category, count in counts.items():
        percent = (count / denom * 100) if denom else 0

        rows.append({
            "Variable": variable_name,
            "Category": category,
            "n": int(count),
            "Percent": round(percent, 1),
            "Total participants": N,
            "Valid responses": valid_n,
            "Missing responses": missing_n,
        })

    # Add an explicit missing row if anything is missing.
    if missing_n > 0:
        missing_percent = (missing_n / N * 100) if N else 0

        rows.append({
            "Variable": variable_name,
            "Category": "Missing",
            "n": missing_n,
            "Percent": round(missing_percent, 1),
            "Total participants": N,
            "Valid responses": valid_n,
            "Missing responses": missing_n,
        })

    return pd.DataFrame(rows)


# ============================================================
# CLEAN SINGLE-RESPONSE DEMOGRAPHIC VARIABLES
# ============================================================

df[AGE_COL] = clean_text(df[AGE_COL])
df[GENDER_COL] = clean_text(df[GENDER_COL])
df[GENDER_OTHER_COL] = clean_text(df[GENDER_OTHER_COL])
df[STUDENT_COL] = clean_text(df[STUDENT_COL])
df[OCCUPATION_COL] = clean_text(df[OCCUPATION_COL])
df[EXERCISE_FREQUENCY_COL] = clean_text(df[EXERCISE_FREQUENCY_COL])
df[EXERCISE_FREQUENCY_OTHER_COL] = clean_text(
    df[EXERCISE_FREQUENCY_OTHER_COL]
)

# Combine "Other" gender text with the main gender response where needed.
df["Gender_Clean"] = df[GENDER_COL]

mask = df["Gender_Clean"].str.lower().eq("other").fillna(False)
df.loc[mask & df[GENDER_OTHER_COL].notna(), "Gender_Clean"] = (
    "Other: " + df.loc[
        mask & df[GENDER_OTHER_COL].notna(),
        GENDER_OTHER_COL
    ]
)

# Combine "Other" exercise frequency text where needed.
df["Exercise_Frequency_Clean"] = df[EXERCISE_FREQUENCY_COL]

mask = (
    df["Exercise_Frequency_Clean"]
    .str.lower()
    .eq("other")
    .fillna(False)
)

df.loc[
    mask & df[EXERCISE_FREQUENCY_OTHER_COL].notna(),
    "Exercise_Frequency_Clean"
] = (
    "Other: "
    + df.loc[
        mask & df[EXERCISE_FREQUENCY_OTHER_COL].notna(),
        EXERCISE_FREQUENCY_OTHER_COL
    ]
)

# ============================================================
# MULTI-SELECT PHYSICAL ACTIVITY DATA
# ============================================================

activity_labels = {}

for col in ACTIVITY_COLS:
    label = subheader.get(col)

    if pd.isna(label):
        label = col

    activity_labels[col] = str(label).strip()

# Create 0/1 indicator variables for every activity.
for col, label in activity_labels.items():
    df[f"Activity__{label}"] = df[col].notna().astype(int)

# Clean free-text "Other" activity responses.
df["Activity_Other_Text"] = clean_text(df[ACTIVITY_OTHER_COL])

# Aggregate activity counts.
activity_rows = []

for col, label in activity_labels.items():
    count = int(df[col].notna().sum())
    percent = (count / N * 100) if N else 0

    activity_rows.append({
        "Variable": "Physical activity type",
        "Category": label,
        "n": count,
        "Percent": round(percent, 1),
        "Total participants": N,
        "Note": "Multi-select: percentages do not need to sum to 100%",
    })

# Count participants who selected/wrote an "Other" activity.
other_activity_count = int(df["Activity_Other_Text"].notna().sum())

activity_rows.append({
    "Variable": "Physical activity type",
    "Category": "Other",
    "n": other_activity_count,
    "Percent": round(
        (other_activity_count / N * 100) if N else 0,
        1
    ),
    "Total participants": N,
    "Note": "Multi-select: percentages do not need to sum to 100%",
})

activity_summary = pd.DataFrame(activity_rows)

# ============================================================
# BUILD ALL AGGREGATE DEMOGRAPHIC TABLES
# ============================================================

summary_tables = [
    frequency_table(df[AGE_COL], "Age range"),
    frequency_table(df["Gender_Clean"], "Gender"),
    frequency_table(df[STUDENT_COL], "Student status"),
    frequency_table(
        df["Exercise_Frequency_Clean"],
        "Exercise frequency"
    ),
    activity_summary,
]

demographic_summary = pd.concat(
    summary_tables,
    ignore_index=True,
    sort=False
)

# ============================================================
# OCCUPATION DATA
# ============================================================

# Occupation is open-ended, so it is usually better to keep the
# responses available rather than force them into arbitrary groups.
occupation_responses = (
    df[[participant_col, OCCUPATION_COL]]
    .rename(columns={OCCUPATION_COL: "Occupation"})
    .sort_values(participant_col)
)

occupation_frequency = frequency_table(
    df[OCCUPATION_COL],
    "Occupation (verbatim response)"
)

# ============================================================
# OTHER FREE-TEXT RESPONSES
# ============================================================

other_responses = df[
    [
        participant_col,
        GENDER_OTHER_COL,
        EXERCISE_FREQUENCY_OTHER_COL,
        "Activity_Other_Text",
    ]
].copy()

other_responses.columns = [
    "Participant ID",
    "Gender - Other",
    "Exercise Frequency - Other",
    "Physical Activity - Other",
]

# Keep only rows with at least one free-text response.
other_responses = other_responses[
    other_responses.iloc[:, 1:].notna().any(axis=1)
]

# ============================================================
# DE-IDENTIFIED PARTICIPANT-LEVEL DATASET
# ============================================================

participant_output = pd.DataFrame({
    "Participant ID": df[participant_col],
    "Age range": df[AGE_COL],
    "Gender": df["Gender_Clean"],
    "Student status": df[STUDENT_COL],
    "Occupation": df[OCCUPATION_COL],
    "Exercise frequency": df["Exercise_Frequency_Clean"],
})

for col, label in activity_labels.items():
    participant_output[label] = df[f"Activity__{label}"]

participant_output["Other activity"] = df["Activity_Other_Text"]

participant_output = participant_output.sort_values("Participant ID")

# ============================================================
# SAVE RESULTS
# ============================================================

demographic_summary.to_csv(
    OUTPUT_DIR / "demographic_summary.csv",
    index=False
)

participant_output.to_csv(
    OUTPUT_DIR / "demographic_participant_level_deidentified.csv",
    index=False
)

occupation_responses.to_csv(
    OUTPUT_DIR / "occupation_responses.csv",
    index=False
)

occupation_frequency.to_csv(
    OUTPUT_DIR / "occupation_frequency.csv",
    index=False
)

other_responses.to_csv(
    OUTPUT_DIR / "other_free_text_responses.csv",
    index=False
)

# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report_lines = []

report_lines.append("DEMOGRAPHIC ANALYSIS")
report_lines.append("=" * 70)
report_lines.append(f"Unique participants analyzed: N = {N}")
report_lines.append("")

if not duplicate_rows.empty:
    duplicate_ids = (
        duplicate_rows[participant_col]
        .drop_duplicates()
        .tolist()
    )

    report_lines.append(
        "Duplicate participant IDs detected and deduplicated: "
        + ", ".join(duplicate_ids)
    )
    report_lines.append("")

for variable in demographic_summary["Variable"].drop_duplicates():

    report_lines.append(variable)
    report_lines.append("-" * len(variable))

    subset = demographic_summary[
        demographic_summary["Variable"] == variable
    ]

    for _, row in subset.iterrows():
        report_lines.append(
            f"{row['Category']}: "
            f"{int(row['n'])} "
            f"({row['Percent']:.1f}%)"
        )

    report_lines.append("")

report_lines.append("NOTE:")
report_lines.append(
    "Physical activity was a multi-select question, so percentages "
    "for activity types do not need to sum to 100%."
)
report_lines.append(
    "Names and email addresses were intentionally excluded from "
    "all analysis outputs."
)

report_text = "\n".join(report_lines)

with open(
    OUTPUT_DIR / "demographic_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print(report_text)

print("\nFiles saved to:")
print(OUTPUT_DIR.resolve())

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {file.name}")
