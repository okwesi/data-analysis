"""
RoSAS Analysis Script
=====================

Primary reference:
Carpinella, C. M., Wyman, A. B., Perez, M. A., & Stroessner, S. J. (2017).
The Robotic Social Attributes Scale (RoSAS): Development and Validation.
Proceedings of the 2017 ACM/IEEE International Conference on Human-Robot
Interaction (HRI '17), 254-262.
DOI: 10.1145/2909824.3020208

STANDARD FULL RoSAS STRUCTURE
-----------------------------
Warmth:
    Organic, Sociable, Emotional, Compassionate, Happy, Feeling

Competence:
    Reliable, Competent, Knowledgeable, Interactive, Responsive, Capable

Discomfort:
    Awkward, Scary, Strange, Awful, Dangerous, Aggressive

Each participant's score for a RoSAS dimension is the MEAN of its six
1-to-9 item ratings.

This script also calculates Cronbach's alpha for each six-item dimension,
which follows the reliability analysis used in the RoSAS validation
literature.

IMPORTANT:
This script does NOT run exploratory factor analysis (EFA) or confirmatory
factor analysis (CFA). With a sample of 17 participants, factor analysis
would not be appropriate. The validated three-factor structure from
Carpinella et al. (2017) is used directly.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Study Data(Post-Study Survey) ROSAS.csv"

OUTPUT_DIR = Path("analysis_results/rosas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VALIDATED RoSAS FACTOR STRUCTURE
# ============================================================

ROSAS_FACTORS = {
    "Warmth": [
        "Organic",
        "Sociable",
        "Emotional",
        "Compassionate",
        "Happy",
        "Feeling",
    ],

    "Competence": [
        "Reliable",
        "Competent",
        "Knowledgeable",
        "Interactive",
        "Responsive",
        "Capable",
    ],

    "Discomfort": [
        "Awkward",
        "Scary",
        "Strange",
        "Awful",
        "Dangerous",
        "Aggressive",
    ],
}

ALL_ITEMS = (
    ROSAS_FACTORS["Warmth"]
    + ROSAS_FACTORS["Competence"]
    + ROSAS_FACTORS["Discomfort"]
)


# ============================================================
# LOAD SURVEYMONKEY CSV
# ============================================================

raw = pd.read_csv(INPUT_FILE)

# SurveyMonkey stores the actual RoSAS adjective names in the
# first data row.
subheader = raw.iloc[0].copy()

# Remove that SurveyMonkey secondary-header row.
df = raw.iloc[1:].copy().reset_index(drop=True)

# Rename exported columns using the adjective names from the
# SurveyMonkey secondary header.
rename_map = {
    raw.columns[0]: "Participant ID"
}

for i in range(1, len(raw.columns)):
    label = subheader.iloc[i]

    if pd.notna(label):
        rename_map[raw.columns[i]] = str(label).strip()

df = df.rename(columns=rename_map)


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

# Save duplicate IDs if present.
duplicates = df[
    df.duplicated("Participant ID", keep=False)
].copy()

if not duplicates.empty:
    duplicates.to_csv(
        OUTPUT_DIR / "rosas_duplicate_participants.csv",
        index=False
    )

# Keep one row per participant.
df = df.drop_duplicates(
    subset="Participant ID",
    keep="first"
).copy()

TOTAL_N = len(df)


# ============================================================
# CONVERT RoSAS RESPONSES TO NUMBERS
# ============================================================

def rosas_to_numeric(value):
    """
    Convert SurveyMonkey RoSAS responses to numeric 1-9 scores.

    SurveyMonkey exported the endpoints as text:
        'Definitely not Associated 1' -> 1
        'Definitely Associated 9'     -> 9

    Middle responses are already stored as 2-8.
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    endpoint_map = {
        "definitely not associated 1": 1.0,
        "definitely associated 9": 9.0,
    }

    if text.lower() in endpoint_map:
        return endpoint_map[text.lower()]

    try:
        number = float(text)

        if 1 <= number <= 9:
            return number

        return np.nan

    except ValueError:
        return np.nan


# Confirm all expected items exist.
missing_columns = [
    item for item in ALL_ITEMS
    if item not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following expected RoSAS items were not found in the CSV:\n"
        + ", ".join(missing_columns)
    )


# Convert every RoSAS item to numeric 1-9.
for item in ALL_ITEMS:
    df[item] = df[item].apply(rosas_to_numeric)


# ============================================================
# MISSING-DATA AUDIT
# ============================================================

missing_rows = []

for _, row in df.iterrows():

    for factor, items in ROSAS_FACTORS.items():

        for item in items:

            if pd.isna(row[item]):
                missing_rows.append({
                    "Participant ID": row["Participant ID"],
                    "Factor": factor,
                    "Item": item,
                })


missing_data = pd.DataFrame(
    missing_rows,
    columns=[
        "Participant ID",
        "Factor",
        "Item",
    ]
)

missing_data.to_csv(
    OUTPUT_DIR / "rosas_missing_data.csv",
    index=False
)


# ============================================================
# PARTICIPANT-LEVEL RoSAS FACTOR SCORES
# ============================================================
#
# Standard scoring:
#
# Factor score = mean of its six items.
#
# Primary analysis uses STRICT six-item scoring:
# if one of the six items is missing, that participant's score
# for that factor is left missing rather than inventing/imputing
# a response.
#
# We also save the number of items answered so the decision is
# completely transparent.

for factor, items in ROSAS_FACTORS.items():

    df[f"{factor} Items Answered"] = (
        df[items]
        .notna()
        .sum(axis=1)
    )

    # STRICT standard six-item mean
    df[f"{factor} Score"] = (
        df[items]
        .mean(axis=1, skipna=False)
    )


# ============================================================
# ITEM-LEVEL DESCRIPTIVE STATISTICS
# ============================================================

item_summary_rows = []

for factor, items in ROSAS_FACTORS.items():

    for item in items:

        values = df[item].dropna()

        item_summary_rows.append({
            "Factor": factor,
            "Item": item,
            "Valid N": int(values.count()),
            "Missing N": int(TOTAL_N - values.count()),
            "Mean": round(float(values.mean()), 3),
            "SD": round(float(values.std(ddof=1)), 3)
                if len(values) > 1 else np.nan,
            "Median": round(float(values.median()), 3),
            "Q1": round(float(values.quantile(0.25)), 3),
            "Q3": round(float(values.quantile(0.75)), 3),
            "Minimum": float(values.min()),
            "Maximum": float(values.max()),
        })


item_summary = pd.DataFrame(item_summary_rows)


# ============================================================
# FACTOR / SUBSCALE DESCRIPTIVE STATISTICS
# ============================================================

factor_summary_rows = []

for factor in ROSAS_FACTORS:

    score_col = f"{factor} Score"

    values = df[score_col].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    factor_summary_rows.append({
        "Factor": factor,
        "Valid N": int(values.count()),
        "Missing N": int(TOTAL_N - values.count()),
        "Mean": round(float(values.mean()), 3),
        "SD": round(float(values.std(ddof=1)), 3)
            if len(values) > 1 else np.nan,
        "Median": round(float(values.median()), 3),
        "Q1": round(float(q1), 3),
        "Q3": round(float(q3), 3),
        "IQR": round(float(q3 - q1), 3),
        "Minimum": round(float(values.min()), 3),
        "Maximum": round(float(values.max()), 3),
    })


factor_summary = pd.DataFrame(factor_summary_rows)


# ============================================================
# CRONBACH'S ALPHA
# ============================================================
#
# alpha = k/(k-1) * [1 - sum(item variances)/variance(total score)]
#
# Complete cases are used within each six-item factor so no
# response values are fabricated.

def cronbach_alpha(data):
    """
    Calculate Cronbach's alpha using complete rows.
    """

    complete = data.dropna(axis=0, how="any").copy()

    n = len(complete)
    k = complete.shape[1]

    if n < 2 or k < 2:
        return np.nan, n

    item_variances = complete.var(
        axis=0,
        ddof=1
    )

    total_scores = complete.sum(axis=1)

    total_variance = total_scores.var(ddof=1)

    if total_variance == 0:
        return np.nan, n

    alpha = (
        k / (k - 1)
        * (
            1
            - item_variances.sum()
            / total_variance
        )
    )

    return float(alpha), n


reliability_rows = []

for factor, items in ROSAS_FACTORS.items():

    alpha, alpha_n = cronbach_alpha(df[items])

    reliability_rows.append({
        "Factor": factor,
        "Items": len(items),
        "Complete-case N": alpha_n,
        "Cronbach Alpha": (
            round(alpha, 3)
            if pd.notna(alpha)
            else np.nan
        ),
    })


reliability_summary = pd.DataFrame(
    reliability_rows
)


# ============================================================
# FACTOR CORRELATIONS
# ============================================================
#
# Some RoSAS validation/use papers also examine relationships
# among Warmth, Competence, and Discomfort.
#
# With N=17 these should be treated as EXPLORATORY, not as a
# central result.
#
# Spearman correlations are used because of the small sample and
# ordinal origin of the 1-9 ratings.

factor_score_cols = [
    "Warmth Score",
    "Competence Score",
    "Discomfort Score",
]

factor_correlations = (
    df[factor_score_cols]
    .corr(method="spearman", min_periods=3)
)


# ============================================================
# PARTICIPANT-LEVEL OUTPUT
# ============================================================

participant_columns = ["Participant ID"]

for factor, items in ROSAS_FACTORS.items():
    participant_columns.extend(items)
    participant_columns.append(
        f"{factor} Items Answered"
    )
    participant_columns.append(
        f"{factor} Score"
    )


participant_scores = (
    df[participant_columns]
    .sort_values("Participant ID")
    .reset_index(drop=True)
)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

participant_scores.to_csv(
    OUTPUT_DIR / "rosas_participant_scores.csv",
    index=False
)

factor_summary.to_csv(
    OUTPUT_DIR / "rosas_factor_summary.csv",
    index=False
)

reliability_summary.to_csv(
    OUTPUT_DIR / "rosas_cronbach_alpha.csv",
    index=False
)

item_summary.to_csv(
    OUTPUT_DIR / "rosas_item_summary.csv",
    index=False
)

factor_correlations.to_csv(
    OUTPUT_DIR / "rosas_factor_correlations_exploratory.csv"
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report = []

report.append("RoSAS ANALYSIS")
report.append("=" * 72)
report.append("")
report.append(
    "Reference: Carpinella et al. (2017), "
    "The Robotic Social Attributes Scale (RoSAS): "
    "Development and Validation."
)
report.append(
    "HRI '17, pp. 254-262. "
    "DOI: 10.1145/2909824.3020208"
)
report.append("")
report.append(
    f"Unique participants in RoSAS dataset: N = {TOTAL_N}"
)
report.append("")

report.append("SCORING")
report.append("-" * 72)
report.append(
    "Each RoSAS factor score is the mean of its six 1-9 item ratings."
)
report.append("")

for factor, items in ROSAS_FACTORS.items():
    report.append(
        f"{factor}: "
        + ", ".join(items)
    )

report.append("")
report.append("FACTOR RESULTS")
report.append("-" * 72)

for _, row in factor_summary.iterrows():

    report.append(
        f"{row['Factor']}: "
        f"N={int(row['Valid N'])}, "
        f"M={row['Mean']:.2f}, "
        f"SD={row['SD']:.2f}, "
        f"Median={row['Median']:.2f}, "
        f"IQR={row['IQR']:.2f}, "
        f"Range={row['Minimum']:.2f}-{row['Maximum']:.2f}"
    )


report.append("")
report.append("INTERNAL CONSISTENCY")
report.append("-" * 72)

for _, row in reliability_summary.iterrows():

    alpha = row["Cronbach Alpha"]

    if pd.isna(alpha):
        alpha_text = "NA"
    else:
        alpha_text = f"{alpha:.3f}"

    report.append(
        f"{row['Factor']}: "
        f"Cronbach's alpha={alpha_text}, "
        f"complete-case N={int(row['Complete-case N'])}"
    )


report.append("")
report.append("MISSING RESPONSES")
report.append("-" * 72)

if missing_data.empty:

    report.append("No missing RoSAS item responses.")

else:

    report.append(
        f"Missing RoSAS item responses: {len(missing_data)}"
    )

    for _, row in missing_data.iterrows():

        report.append(
            f"  {row['Participant ID']}: "
            f"{row['Item']} ({row['Factor']})"
        )


report.append("")
report.append("ANALYSIS NOTE")
report.append("-" * 72)
report.append(
    "The validated three-factor structure was used directly. "
    "No EFA/CFA was performed because the current sample size "
    "is too small for a defensible factor analysis."
)
report.append(
    "The factor-correlation CSV is exploratory and does not need "
    "to be included in the main paper."
)

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "rosas_analysis_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)


# ============================================================
# PRINT RESULTS
# ============================================================

print(report_text)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {file.name}")
