import pandas as pd
import numpy as np
from pathlib import Path

try:
    from scipy.stats import wilcoxon, rankdata
except ImportError:
    raise SystemExit(
        "This script requires scipy. Install it with:\n"
        "pip install pandas scipy"
    )

# ============================================================
# CONFIGURATION
# ============================================================

PRE_FILE = "Study Data(Pre-Study Survey).csv"
POST_FILE = "Study Data(Post-Study Survey) (1).csv"

OUTPUT_DIR = Path("analysis_results/sam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SAM SCORING
# ============================================================

AROUSAL_ORDER = [
    "Very Calm",
    "Calm",
    "Neutral",
    "Excited",
    "Very Excited",
]

VALENCE_ORDER = [
    "Very Unpleasant",
    "Unpleasant",
    "Neutral",
    "Pleasant",
    "Very Pleasant",
]

AROUSAL_SCORE = {label: i + 1 for i, label in enumerate(AROUSAL_ORDER)}
VALENCE_SCORE = {label: i + 1 for i, label in enumerate(VALENCE_ORDER)}

# 0 is used only for missing PRE-STUDY responses so all 17
# post-study participants remain in the participant-level analysis.
#
# IMPORTANT:
# 0 is not an actual SAM response category.

# ============================================================
# LOAD DATA
# ============================================================

pre_raw = pd.read_csv(PRE_FILE)
post_raw = pd.read_csv(POST_FILE)


def remove_surveymonkey_subheader(df):
    df = df.copy()
    first_col = df.columns[0]

    mask = (
        df[first_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("open-ended response")
    )

    return df.loc[~mask].copy().reset_index(drop=True)


pre = remove_surveymonkey_subheader(pre_raw)
post = remove_surveymonkey_subheader(post_raw)

# ============================================================
# IDENTIFY COLUMNS
# ============================================================

PRE_ID = pre.columns[0]
PRE_AROUSAL = pre.columns[1]
PRE_VALENCE = pre.columns[2]

POST_ID = post.columns[0]
POST_AROUSAL = post.columns[1]
POST_VALENCE = post.columns[2]

# ============================================================
# CLEAN DATA
# ============================================================

def clean_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA})
    )


def clean_response(series):
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA})
    )


pre[PRE_ID] = clean_id(pre[PRE_ID])
post[POST_ID] = clean_id(post[POST_ID])

pre[PRE_AROUSAL] = clean_response(pre[PRE_AROUSAL])
pre[PRE_VALENCE] = clean_response(pre[PRE_VALENCE])

post[POST_AROUSAL] = clean_response(post[POST_AROUSAL])
post[POST_VALENCE] = clean_response(post[POST_VALENCE])

pre = pre[pre[PRE_ID].notna()].copy()
post = post[post[POST_ID].notna()].copy()

# Save duplicate records if any
pre_duplicates = pre[pre.duplicated(PRE_ID, keep=False)].copy()
post_duplicates = post[post.duplicated(POST_ID, keep=False)].copy()

if not pre_duplicates.empty:
    pre_duplicates.to_csv(
        OUTPUT_DIR / "pre_duplicate_participant_records.csv",
        index=False
    )

if not post_duplicates.empty:
    post_duplicates.to_csv(
        OUTPUT_DIR / "post_duplicate_participant_records.csv",
        index=False
    )

pre = pre.drop_duplicates(PRE_ID, keep="first").copy()
post = post.drop_duplicates(POST_ID, keep="first").copy()

# ============================================================
# RENAME COLUMNS
# ============================================================

pre_clean = pre[[PRE_ID, PRE_AROUSAL, PRE_VALENCE]].copy()
pre_clean.columns = [
    "Participant ID",
    "Pre Arousal Label",
    "Pre Valence Label",
]

post_clean = post[[POST_ID, POST_AROUSAL, POST_VALENCE]].copy()
post_clean.columns = [
    "Participant ID",
    "Post Arousal Label",
    "Post Valence Label",
]

# ============================================================
# KEEP ALL POST-STUDY PARTICIPANTS
# ============================================================

sam = post_clean.merge(
    pre_clean,
    on="Participant ID",
    how="left",
    validate="one_to_one",
)

sam = sam[
    [
        "Participant ID",
        "Pre Arousal Label",
        "Post Arousal Label",
        "Pre Valence Label",
        "Post Valence Label",
    ]
].copy()

sam["Has Pre-Study SAM"] = (
    sam["Pre Arousal Label"].notna()
    & sam["Pre Valence Label"].notna()
)

# ============================================================
# SCORE RESPONSES
# ============================================================

sam["Pre Arousal Score"] = sam["Pre Arousal Label"].map(AROUSAL_SCORE)
sam["Post Arousal Score"] = sam["Post Arousal Label"].map(AROUSAL_SCORE)

sam["Pre Valence Score"] = sam["Pre Valence Label"].map(VALENCE_SCORE)
sam["Post Valence Score"] = sam["Post Valence Label"].map(VALENCE_SCORE)

# Missing PRE values become zero.
sam["Pre Arousal Score"] = (
    sam["Pre Arousal Score"]
    .fillna(0)
    .astype(int)
)

sam["Pre Valence Score"] = (
    sam["Pre Valence Score"]
    .fillna(0)
    .astype(int)
)

sam["Post Arousal Score"] = (
    pd.to_numeric(sam["Post Arousal Score"], errors="coerce")
)

sam["Post Valence Score"] = (
    pd.to_numeric(sam["Post Valence Score"], errors="coerce")
)

sam["Pre Arousal Label"] = (
    sam["Pre Arousal Label"]
    .fillna("Missing Pre-Study")
)

sam["Pre Valence Label"] = (
    sam["Pre Valence Label"]
    .fillna("Missing Pre-Study")
)

# ============================================================
# PRE -> POST CHANGE SCORES FOR ALL 17
# ============================================================
#
# THIS IS THE IMPORTANT PART:
#
# Change = Post - Pre
#
# For participants with missing pre-study data:
# Pre = 0
# Therefore their change is calculated from 0 to their post score.
#
# Example:
# Pre = 0
# Post = 4
# Change = +4
#
# We keep a flag showing whether that change came from a genuine
# pre-study response or from a zero placeholder.

sam["Arousal Change"] = (
    sam["Post Arousal Score"] - sam["Pre Arousal Score"]
)

sam["Valence Change"] = (
    sam["Post Valence Score"] - sam["Pre Valence Score"]
)

sam["Arousal Absolute Change"] = sam["Arousal Change"].abs()
sam["Valence Absolute Change"] = sam["Valence Change"].abs()


def arousal_change_label(change):
    if pd.isna(change):
        return "Missing Post-Study"
    if change > 0:
        return "More excited"
    if change < 0:
        return "Calmer"
    return "No change"


def valence_change_label(change):
    if pd.isna(change):
        return "Missing Post-Study"
    if change > 0:
        return "More pleasant"
    if change < 0:
        return "Less pleasant"
    return "No change"


sam["Arousal Change Direction"] = (
    sam["Arousal Change"].apply(arousal_change_label)
)

sam["Valence Change Direction"] = (
    sam["Valence Change"].apply(valence_change_label)
)

# Explicit flag so you can see which change scores are based on 0 placeholders.
sam["Change Score Source"] = np.where(
    sam["Has Pre-Study SAM"],
    "Real Pre + Post",
    "Pre Missing -> Set to 0",
)

sam = sam.sort_values("Participant ID").reset_index(drop=True)

# ============================================================
# ALL-17 PARTICIPANT CHANGE TABLE
# ============================================================

participant_change_table = sam[
    [
        "Participant ID",
        "Has Pre-Study SAM",
        "Change Score Source",

        "Pre Arousal Label",
        "Pre Arousal Score",
        "Post Arousal Label",
        "Post Arousal Score",
        "Arousal Change",
        "Arousal Absolute Change",
        "Arousal Change Direction",

        "Pre Valence Label",
        "Pre Valence Score",
        "Post Valence Label",
        "Post Valence Score",
        "Valence Change",
        "Valence Absolute Change",
        "Valence Change Direction",
    ]
].copy()

# ============================================================
# CHANGE SUMMARY FOR ALL 17
# ============================================================

def summarize_change(series, construct):
    s = pd.to_numeric(series, errors="coerce").dropna()

    return {
        "Construct": construct,
        "N": len(s),
        "Mean Change": round(float(s.mean()), 3),
        "Median Change": round(float(s.median()), 3),
        "Minimum Change": float(s.min()),
        "Maximum Change": float(s.max()),
        "Mean Absolute Change": round(float(s.abs().mean()), 3),
    }


change_score_summary = pd.DataFrame(
    [
        summarize_change(
            sam["Arousal Change"],
            "Arousal"
        ),
        summarize_change(
            sam["Valence Change"],
            "Valence"
        ),
    ]
)

# ============================================================
# CHANGE-DIRECTION COUNTS FOR ALL 17
# ============================================================

def direction_summary(series, construct):
    counts = series.value_counts()
    total = len(series)

    rows = []

    for category, count in counts.items():
        rows.append({
            "Construct": construct,
            "Direction": category,
            "n": int(count),
            "Percent": round(count / total * 100, 1),
        })

    return pd.DataFrame(rows)


change_direction_summary = pd.concat(
    [
        direction_summary(
            sam["Arousal Change Direction"],
            "Arousal"
        ),
        direction_summary(
            sam["Valence Change Direction"],
            "Valence"
        ),
    ],
    ignore_index=True,
)

# ============================================================
# DESCRIPTIVE PRE/POST SUMMARY FOR ALL 17
# ============================================================

def descriptive_all17(series, construct, timepoint):
    s = pd.to_numeric(series, errors="coerce").dropna()

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)

    return {
        "Construct": construct,
        "Timepoint": timepoint,
        "N": len(s),
        "Mean": round(float(s.mean()), 3),
        "SD": round(float(s.std(ddof=1)), 3) if len(s) > 1 else np.nan,
        "Median": round(float(s.median()), 3),
        "Q1": round(float(q1), 3),
        "Q3": round(float(q3), 3),
        "IQR": round(float(q3 - q1), 3),
        "Minimum": float(s.min()),
        "Maximum": float(s.max()),
    }


all17_descriptive = pd.DataFrame(
    [
        descriptive_all17(
            sam["Pre Arousal Score"],
            "Arousal",
            "Pre"
        ),
        descriptive_all17(
            sam["Post Arousal Score"],
            "Arousal",
            "Post"
        ),
        descriptive_all17(
            sam["Pre Valence Score"],
            "Valence",
            "Pre"
        ),
        descriptive_all17(
            sam["Post Valence Score"],
            "Valence",
            "Post"
        ),
    ]
)

# ============================================================
# VALID PAIRED TEST ONLY
# ============================================================
#
# Statistical test still excludes artificial zero placeholders.
# This keeps the significance test valid while preserving all 17
# participants in the change-score dataset.

paired = sam[sam["Has Pre-Study SAM"]].copy()


def rank_biserial_effect(pre_scores, post_scores):
    pre_scores = pd.to_numeric(pre_scores, errors="coerce")
    post_scores = pd.to_numeric(post_scores, errors="coerce")

    valid = pre_scores.notna() & post_scores.notna()
    diff = (post_scores[valid] - pre_scores[valid]).to_numpy()

    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    ranks = rankdata(np.abs(diff))
    positive_rank_sum = ranks[diff > 0].sum()
    negative_rank_sum = ranks[diff < 0].sum()

    return float(
        (positive_rank_sum - negative_rank_sum) / ranks.sum()
    )


def wilcoxon_analysis(pre_scores, post_scores, construct):
    pre_scores = pd.to_numeric(pre_scores, errors="coerce")
    post_scores = pd.to_numeric(post_scores, errors="coerce")

    valid = pre_scores.notna() & post_scores.notna()

    pre_valid = pre_scores[valid]
    post_valid = post_scores[valid]

    differences = post_valid - pre_valid
    nonzero_n = int((differences != 0).sum())
    zero_n = int((differences == 0).sum())

    if nonzero_n == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(
            post_valid,
            pre_valid,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )

        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    effect = rank_biserial_effect(pre_valid, post_valid)

    return {
        "Construct": construct,
        "Total participants in dataset": len(sam),
        "Valid paired N used in Wilcoxon": int(len(pre_valid)),
        "Zero-placeholder participants excluded from test":
            int((~sam["Has Pre-Study SAM"]).sum()),
        "Wilcoxon statistic": round(statistic, 4),
        "p-value": round(p_value, 6),
        "Rank-biserial effect": round(effect, 4),
        "Ties": zero_n,
        "Non-zero changes": nonzero_n,
    }


wilcoxon_results = pd.DataFrame(
    [
        wilcoxon_analysis(
            paired["Pre Arousal Score"],
            paired["Post Arousal Score"],
            "Arousal",
        ),
        wilcoxon_analysis(
            paired["Pre Valence Score"],
            paired["Post Valence Score"],
            "Valence",
        ),
    ]
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

participant_change_table.to_csv(
    OUTPUT_DIR / "sam_pre_post_change_all17.csv",
    index=False
)

sam.to_csv(
    OUTPUT_DIR / "sam_all17_full_data.csv",
    index=False
)

change_score_summary.to_csv(
    OUTPUT_DIR / "sam_change_score_summary_all17.csv",
    index=False
)

change_direction_summary.to_csv(
    OUTPUT_DIR / "sam_change_direction_summary_all17.csv",
    index=False
)

all17_descriptive.to_csv(
    OUTPUT_DIR / "sam_pre_post_descriptive_all17.csv",
    index=False
)

paired.to_csv(
    OUTPUT_DIR / "sam_valid_real_pre_post_pairs.csv",
    index=False
)

wilcoxon_results.to_csv(
    OUTPUT_DIR / "sam_wilcoxon_valid_pairs.csv",
    index=False
)

# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

missing_pre_ids = sam.loc[
    ~sam["Has Pre-Study SAM"],
    "Participant ID"
].tolist()

report = []

report.append("SAM PRE -> POST CHANGE ANALYSIS")
report.append("=" * 72)
report.append("")
report.append(f"All post-study participants included: N = {len(sam)}")
report.append(f"Real pre/post pairs: N = {len(paired)}")
report.append(
    f"Missing pre-study participants assigned Pre=0: "
    f"N = {len(missing_pre_ids)}"
)
report.append(
    "Participants assigned Pre=0: "
    + (", ".join(missing_pre_ids) if missing_pre_ids else "None")
)
report.append("")

report.append("IMPORTANT CHANGE CALCULATION")
report.append("-" * 72)
report.append("Change Score = Post Score - Pre Score")
report.append("")
report.append(
    "For missing pre-study participants, Pre Score = 0, "
    "so their change is calculated from 0 to their post-study score."
)
report.append("")

report.append("CHANGE SCORE SUMMARY - ALL 17")
report.append("-" * 72)

for _, row in change_score_summary.iterrows():
    report.append(
        f"{row['Construct']}: "
        f"N={int(row['N'])}, "
        f"mean change={row['Mean Change']:.2f}, "
        f"median change={row['Median Change']:.2f}, "
        f"mean absolute change={row['Mean Absolute Change']:.2f}, "
        f"range={row['Minimum Change']:.0f} to "
        f"{row['Maximum Change']:.0f}"
    )

report.append("")
report.append("CHANGE DIRECTIONS - ALL 17")
report.append("-" * 72)

for construct in ["Arousal", "Valence"]:
    report.append(construct)

    subset = change_direction_summary[
        change_direction_summary["Construct"] == construct
    ]

    for _, row in subset.iterrows():
        report.append(
            f"  {row['Direction']}: "
            f"{int(row['n'])} ({row['Percent']:.1f}%)"
        )

report.append("")
report.append("VALID WILCOXON TESTS")
report.append("-" * 72)
report.append(
    "The inferential test uses only genuine pre/post pairs. "
    "Participants whose pre score was artificially set to 0 are "
    "excluded from the Wilcoxon test."
)

for _, row in wilcoxon_results.iterrows():
    report.append(
        f"{row['Construct']}: "
        f"paired N={int(row['Valid paired N used in Wilcoxon'])}, "
        f"W={row['Wilcoxon statistic']:.4f}, "
        f"p={row['p-value']:.6f}, "
        f"rank-biserial effect={row['Rank-biserial effect']:.4f}"
    )

report_text = "\n".join(report)

with open(
    OUTPUT_DIR / "sam_analysis_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)

print(report_text)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {path.name}")
