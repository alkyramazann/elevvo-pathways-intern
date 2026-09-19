import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

RANDOM_STATE = 42
TEST_SIZE = 0.2
POLY_DEGREES = [2, 3]  
CV_FOLDS = 5
BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#9AA5B1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHART_DIR = PROJECT_ROOT / "charts"
REPORT_DIR = PROJECT_ROOT / "reports"
for folder in (DATA_DIR, CHART_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
ORIGINAL_PATH = DATA_DIR / "student_scores_original.csv"
CLEANED_PATH = DATA_DIR / "student_scores_cleaned.csv"

LOG_LINES = []


def log(message=""):
    """Print a line and keep it for reports/analysis_summary.txt."""
    print(message)
    LOG_LINES.append(str(message))


def section(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


def pretty(column):
    """Hours_Studied -> Hours Studied"""
    return column.replace("_", " ")


def save_fig(fig, filename):
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved charts/{filename}")



def load_dataset(path):
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\nPut the CSV in the data/ folder as "
                 f"'{path.name}' and run the script again.")
    return pd.read_csv(path)


df_raw = load_dataset(ORIGINAL_PATH)



def inspect_dataset(df):
    section("3. DATASET INSPECTION")
    log(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    log("\nColumn names and data types:")
    log(df.dtypes.astype(str).to_string())

    missing = df.isna().sum()
    log("\nMissing values (columns that have any):")
    log(missing[missing > 0].to_string() if missing.any() else "  none")
    log(f"\nDuplicate rows: {int(df.duplicated().sum())}")

    categorical = df.select_dtypes(exclude="number").columns
    log("\nUnique values of categorical columns:")
    for col in categorical:
        log(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")

    numeric = df.select_dtypes(include="number").columns
    log("\nNumerical ranges (min - max):")
    for col in numeric:
        log(f"  {col}: {df[col].min()} - {df[col].max()}")

    log("\nDescriptive statistics (numeric columns):")
    log(df[numeric].describe().T.round(3).to_string())


def find_column(columns, patterns, exclude=()):
    """Return the first column whose (lower-cased) name contains all keywords of a pattern."""
    for pattern in patterns:
        for col in columns:
            name = col.lower().replace("_", " ")
            if all(k in name for k in pattern) and not any(k in name for k in exclude):
                return col
    return None


def detect_roles(df):
    """Identify the target and the columns most useful as predictors, using column names."""
    cols = list(df.columns)
    roles = {
        "target": find_column(
            cols,
            [("exam", "score"), ("final", "score"), ("final", "exam"), ("score",)],
            exclude=("previous", "prior", "past"),
        ),
        "study_hours": find_column(cols, [("stud", "hour"), ("hour",)], exclude=("sleep",)),
        "sleep": find_column(cols, [("sleep",)]),
        "participation": find_column(cols, [("particip",)]),
        "attendance": find_column(cols, [("attend",)]),
        "previous_score": find_column(
            cols, [("previous", "score"), ("prior", "score"), ("past", "score"), ("previous",)]
        ),
    }
    if roles["target"] is None or roles["study_hours"] is None:
        sys.exit("Could not identify the target (exam score) and/or study-hours column. "
                 f"Columns found: {cols}")
    return roles


inspect_dataset(df_raw)
ROLES = detect_roles(df_raw)
TARGET, HOURS = ROLES["target"], ROLES["study_hours"]

log("\nColumn roles identified from the dataset:")
for role, col in ROLES.items():
    log(f"  {role:<15} -> {col if col else 'NOT PRESENT in this dataset'}")



def iqr_fences(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def clean_data(df, target):
    section("4. DATA CLEANING")
    df = df.copy()
    df.columns = df.columns.str.strip()
    n_start = len(df)

    converted_any = False
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(df[col].notna().sum(), 1) > 0.95:
                df[col] = converted
                converted_any = True
                log(f"[types] '{col}' was stored as text but is numeric -> converted")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    log(f"[types] {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns"
        + ("" if converted_any else "; all columns already have the correct data type"))

    changed = False
    for col in categorical_cols:
        s = df[col].astype("string").str.strip()
        spellings = {}
        for value in s.dropna().unique():
            spellings.setdefault(value.lower(), []).append(value)
        counts = s.value_counts()
        mapping = {v: max(vs, key=lambda x: counts[x])
                   for vs in spellings.values() if len(vs) > 1 for v in vs}
        if mapping:
            s = s.replace(mapping)
            log(f"[categories] '{col}': merged inconsistent spellings {mapping}")
            changed = True
        df[col] = s.astype(object).where(s.notna(), np.nan)
    if not changed:
        log("[categories] no inconsistent spellings/whitespace found in categorical columns")

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log(f"[duplicates] {n_dup} duplicate rows found" + (" and removed" if n_dup else " - nothing to remove"))

    n_missing_target = int(df[target].isna().sum())
    df = df.dropna(subset=[target])
    log(f"[missing target] {n_missing_target} rows without '{target}'"
        + (" removed" if n_missing_target else " - none"))

    percent_like = [c for c in numeric_cols if any(k in c.lower() for k in ("score", "attend", "percent"))]
    invalid = pd.Series(False, index=df.index)
    found_any = False
    for col in numeric_cols:
        rules = [("negative value", df[col] < 0)]
        if col in percent_like:
            rules.append(("above 100 (valid range is 0-100)", df[col] > 100))
        if "sleep" in col.lower():
            rules.append(("more than 24 hours", df[col] > 24))
        for description, mask in rules:
            mask = mask.fillna(False)
            if mask.any():
                found_any = True
                log(f"[invalid] '{col}': {int(mask.sum())} row(s) with {description}; values = "
                    f"{sorted(df.loc[mask, col].unique().tolist())}")
                invalid |= mask
    if found_any:
        n_invalid = int(invalid.sum())
        df = df.loc[~invalid]
        log(f"[invalid] removed {n_invalid} row(s). Reason: the true value is unknown, so the "
            "value cannot be corrected, and the loss is negligible.")
    else:
        log("[invalid] no impossible values found")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    for col, n in missing.items():
        if col in categorical_cols:
            df[col] = df[col].fillna("Unknown")
            log(f"[missing] '{col}': {n} missing values ({n / len(df):.1%}) -> labelled 'Unknown'. "
                "This uses no statistics from the data, so nothing can leak from test to train, "
                "and it keeps the rows.")
        else:
            log(f"[missing] '{col}': {n} missing numeric values kept as NaN; the model pipelines "
                "impute them with the TRAINING-set median only.")
    if missing.empty:
        log("[missing] no missing values remain")

    log(f"\nRows: {n_start:,} -> {len(df):,} ({n_start - len(df)} removed)")
    return df.reset_index(drop=True), numeric_cols, categorical_cols


def investigate_outliers(df, numeric_cols, target, hours):
    """Outliers are investigated, NOT removed automatically."""
    log("\n[outliers] IQR rule (1.5 x IQR) applied to every numeric column:")
    rows = []
    flags = {}
    for col in numeric_cols:
        low, high = iqr_fences(df[col])
        mask = (df[col] < low) | (df[col] > high)
        flags[col] = mask
        rows.append({"Column": col, "Lower fence": round(low, 2), "Upper fence": round(high, 2),
                     "Outliers": int(mask.sum()), "Share": f"{mask.mean():.2%}",
                     "Min": df[col].min(), "Max": df[col].max()})
    log(pd.DataFrame(rows).to_string(index=False))

    t_mask = flags[target]
    others = [c for c in numeric_cols if c != target]
    if t_mask.any():
        log(f"\n[outliers] {int(t_mask.sum())} students ({t_mask.mean():.2%}) have unusually high/low '{target}' "
            f"(values {sorted(df.loc[t_mask, target].unique().tolist())}).")
        compare = pd.DataFrame({"outlier rows": df.loc[t_mask, others].mean(),
                                "other rows": df.loc[~t_mask, others].mean()}).round(2)
        log("  Mean of the other numeric features, outlier rows vs. the rest:")
        log(compare.to_string())
        smd = ((df.loc[t_mask, others].mean() - df.loc[~t_mask, others].mean()) / df[others].std()).round(2)
        biggest = smd.abs().idxmax()
        log(f"  Largest difference: {smd[biggest]:+.2f} standard deviations ({biggest}).")
        if smd.abs().max() < 0.5:
            log("  -> The recorded numeric features do NOT explain why these students scored unusually "
                "high/low: their study hours, attendance, etc. look like everyone else's.")
        else:
            log("  -> At least one recorded feature differs noticeably for these students, "
                "so part of the unusual score may be explainable.")
    h_mask = flags[hours]
    if h_mask.any():
        log(f"\n[outliers] {int(h_mask.sum())} students have extreme '{hours}' "
            f"(max = {df[hours].max()}). Their mean {target} is {df.loc[h_mask, target].mean():.2f} "
            f"vs {df.loc[~h_mask, target].mean():.2f} for the rest, which is consistent with a "
            "plausible (not erroneous) value.")

    any_flag = t_mask | h_mask
    r_all = df[hours].corr(df[target])
    r_wo = df.loc[~any_flag, hours].corr(df.loc[~any_flag, target])
    log(f"\n[outliers] Sensitivity check - correlation({hours}, {target}): "
        f"{r_all:.3f} with all rows vs {r_wo:.3f} without the flagged rows.")
    log("[outliers] DECISION: RETAIN them in the main analysis. (1) Every flagged value is inside the "
        "valid range (impossible values were already removed in step 4.5), so none is provably an error. "
        "(2) Deleting the rows a model predicts badly would be circular and would make the results look "
        "better than they are. (3) They may still distort the error metrics, so section 11 measures "
        "exactly how much of the prediction error they cause.")


df_clean, NUMERIC_COLS, CATEGORICAL_COLS = clean_data(df_raw, TARGET)
investigate_outliers(df_clean, NUMERIC_COLS, TARGET, HOURS)
df_clean.to_csv(CLEANED_PATH, index=False)
log(f"\nSaved cleaned dataset -> data/{CLEANED_PATH.name} ({len(df_clean):,} rows)")


section("5. EXPLORATORY DATA ANALYSIS")
FEATURE_NUMERIC = [c for c in NUMERIC_COLS if c != TARGET]
corr_with_target = df_clean[NUMERIC_COLS].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
log(f"Target '{TARGET}': mean={df_clean[TARGET].mean():.2f}, median={df_clean[TARGET].median():.1f}, "
    f"std={df_clean[TARGET].std():.2f}, min={df_clean[TARGET].min()}, max={df_clean[TARGET].max()}, "
    f"skew={df_clean[TARGET].skew():.2f}")
log(f"Study hours '{HOURS}': mean={df_clean[HOURS].mean():.2f}, std={df_clean[HOURS].std():.2f}, "
    f"min={df_clean[HOURS].min()}, max={df_clean[HOURS].max()}")
log("\nPearson correlation of each numeric feature with the exam score:")
log(corr_with_target.round(3).to_string())

group_gap = {}
for col in CATEGORICAL_COLS:
    means = df_clean.groupby(col)[TARGET].mean()
    group_gap[col] = means.max() - means.min()
log("\nCategorical features: mean exam score per group (ranked by gap between best and worst group):")
for col in sorted(group_gap, key=group_gap.get, reverse=True):
    means = df_clean.groupby(col)[TARGET].mean().sort_values(ascending=False).round(2)
    log(f"  {col:<28} gap={group_gap[col]:.2f}  {means.to_dict()}")

hours_bins = df_clean.groupby(pd.cut(df_clean[HOURS], [0, 10, 15, 20, 25, 30, 100]),
                              observed=True)[TARGET].agg(["mean", "count"]).round(2)
log("\nMean exam score by study-hours band:")
log(hours_bins.to_string())


def plot_score_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lo, hi = int(df[TARGET].min()), int(df[TARGET].max())
    sns.histplot(df[TARGET], bins=np.arange(lo - 0.5, hi + 1.5, 1), kde=True, color=BLUE,
                 edgecolor="white", ax=ax)
    ax.axvline(df[TARGET].mean(), color=RED, ls="--", lw=2, label=f"Mean = {df[TARGET].mean():.2f}")
    ax.axvline(df[TARGET].median(), color=GREEN, ls=":", lw=2, label=f"Median = {df[TARGET].median():.0f}")
    ax.set(title="Distribution of Final Exam Scores", xlabel=pretty(TARGET), ylabel="Number of students")
    ax.legend()
    save_fig(fig, "score_distribution.png")


def plot_hours_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(df[HOURS], bins=range(int(df[HOURS].min()), int(df[HOURS].max()) + 2),
                 kde=True, color=GREEN, edgecolor="white", ax=ax)
    ax.axvline(df[HOURS].mean(), color=RED, ls="--", lw=2, label=f"Mean = {df[HOURS].mean():.1f}")
    ax.set(title="Distribution of Study Hours", xlabel=pretty(HOURS), ylabel="Number of students")
    ax.legend()
    save_fig(fig, "study_hours_distribution.png")


def plot_hours_vs_score(df):
    rng = np.random.default_rng(RANDOM_STATE)
    r = df[HOURS].corr(df[TARGET])
    means = df.groupby(HOURS)[TARGET].agg(["mean", "count"])
    means = means[means["count"] >= 5]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[HOURS] + rng.uniform(-0.25, 0.25, len(df)), df[TARGET] + rng.uniform(-0.25, 0.25, len(df)),
               s=12, alpha=0.25, color=BLUE, label="Students (jittered slightly)")
    ax.plot(means.index, means["mean"], "-o", color=RED, ms=4, lw=2,
            label="Mean score per study-hours value (n >= 5)")
    ax.set(title=f"Study Hours vs. Final Exam Score (r = {r:.2f})", xlabel=pretty(HOURS), ylabel=pretty(TARGET))
    ax.legend(loc="lower right")
    save_fig(fig, "study_hours_vs_score.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(df[NUMERIC_COLS].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1,
                vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    save_fig(fig, "correlation_heatmap.png")


def plot_numeric_vs_score(df):
    n = len(FEATURE_NUMERIC)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, FEATURE_NUMERIC):
        sns.regplot(x=df[col], y=df[TARGET], ci=None, x_jitter=0.2 if df[col].nunique() <= 60 else 0,
                    scatter_kws={"s": 8, "alpha": 0.2, "color": BLUE}, line_kws={"color": RED}, ax=ax)
        ax.set(title=f"{pretty(col)} (r = {df[col].corr(df[TARGET]):.2f})", xlabel=pretty(col),
               ylabel=pretty(TARGET))
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Numeric Features vs. Final Exam Score", fontsize=15, y=1.01)
    fig.tight_layout()
    save_fig(fig, "numeric_features_vs_score.png")


def plot_categorical_vs_score(df):
    top = sorted(group_gap, key=group_gap.get, reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), top):
        order = df.groupby(col)[TARGET].mean().sort_values().index
        sns.boxplot(data=df, x=col, y=TARGET, order=order, color="#8CB4D8", fliersize=2, ax=ax)
        ax.set(title=f"{pretty(col)} (gap in means = {group_gap[col]:.2f})", xlabel="", ylabel=pretty(TARGET))
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    fig.suptitle("Categorical Features with the Largest Differences in Mean Exam Score", fontsize=15, y=1.01)
    fig.tight_layout()
    save_fig(fig, "categorical_features_vs_score.png")


def plot_outlier_boxplots(df):
    n = len(NUMERIC_COLS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.boxplot(x=df[col], color="#8CB4D8", fliersize=3, ax=ax)
        ax.set_title(pretty(col))
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Outlier Investigation (box = IQR, dots = beyond 1.5 x IQR)", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "outlier_boxplots.png")


log("\nCreating EDA charts:")
plot_score_distribution(df_clean)
plot_hours_distribution(df_clean)
plot_hours_vs_score(df_clean)
plot_correlation_heatmap(df_clean)
plot_numeric_vs_score(df_clean)
plot_categorical_vs_score(df_clean)
plot_outlier_boxplots(df_clean)


section("6. FEATURE SELECTION")
log(f"Target variable : {TARGET}")
log(f"Main feature    : {HOURS}")

WHY = {
    "study_hours": "more time spent studying should mean more practice and better retention",
    "sleep": "rest is linked to concentration and memory consolidation",
    "attendance": "students who attend more are exposed to more instruction (used as the engagement "
                  "measure because this dataset has no 'participation' column)",
    "participation": "active participation reflects engagement with the material",
    "previous_score": "earlier results are a proxy for prior knowledge and ability",
}
reason_by_column = {ROLES[k]: v for k, v in WHY.items() if ROLES.get(k)}
reason_by_column.update({c: "extra instruction outside class could add to what students learn"
                         for c in FEATURE_NUMERIC if "tutor" in c.lower()})
reason_by_column.update({c: "general well-being / lifestyle balance may affect performance"
                         for c in FEATURE_NUMERIC if "physical" in c.lower()})
log("\nNumeric candidate features:")
for col in FEATURE_NUMERIC:
    log(f"  - {col} (r = {corr_with_target[col]:+.2f}): "
        f"{reason_by_column.get(col, 'numeric variable available in the dataset; tested in the all-features model')}")
log("\nCategorical candidate features (one-hot encoded in the full model): "
    "background and context variables (family, school, motivation, resources, ...):")
for col in CATEGORICAL_COLS:
    log(f"  - {col}: {df_clean[col].nunique()} groups")

feature_corr = df_clean[FEATURE_NUMERIC].corr().abs().where(~np.eye(len(FEATURE_NUMERIC), dtype=bool))
log(f"\nLargest absolute correlation between two numeric predictors: {np.nanmax(feature_corr.values):.2f} "
    "(low -> little multicollinearity, so coefficients are reasonably stable)")


def build_feature_sets():
    """Feature combinations built only from columns that exist in this dataset."""
    sleep, previous = ROLES["sleep"], ROLES["previous_score"]
    engagement = [c for c in (ROLES["participation"], ROLES["attendance"]) if c]
    combos = [([HOURS], pretty(HOURS))]
    if sleep:
        combos.append(([HOURS, sleep], f"{pretty(HOURS)} + {pretty(sleep)}"))
    for eng in engagement:
        combos.append(([HOURS, eng], f"{pretty(HOURS)} + {pretty(eng)}"))
    if sleep and engagement:
        combos.append(([HOURS, sleep, engagement[0]],
                       f"{pretty(HOURS)} + {pretty(sleep)} + {pretty(engagement[0])}"))
    if engagement and previous:
        combos.append(([HOURS, engagement[0], previous],
                       f"{pretty(HOURS)} + {pretty(engagement[0])} + {pretty(previous)}"))
    if len(FEATURE_NUMERIC) > 1:
        combos.append((list(FEATURE_NUMERIC), f"All numeric features ({len(FEATURE_NUMERIC)})"))
    if CATEGORICAL_COLS:
        combos.append((list(FEATURE_NUMERIC) + list(CATEGORICAL_COLS),
                       f"All numeric + categorical ({len(FEATURE_NUMERIC) + len(CATEGORICAL_COLS)})"))
    named, seen = [], set()
    for letter, (cols, desc) in zip("ABCDEFGHIJ", combos):
        if tuple(cols) in seen:
            continue
        seen.add(tuple(cols))
        named.append((letter, cols, desc))
    return named


FEATURE_SETS = build_feature_sets()
log("\nFeature combinations that will be tested (Linear Regression):")
for letter, cols, desc in FEATURE_SETS:
    log(f"  Model {letter}: {desc}")
if not ROLES["participation"]:
    if ROLES["attendance"]:
        log("  Note: there is no 'participation' column, so the 'Study Hours + Participation' "
            f"combinations use '{ROLES['attendance']}' as the engagement measure instead.")
    else:
        log("  Note: the dataset has no participation or attendance column, so those combinations are skipped.")


section("7. TRAIN/TEST SPLIT")
ALL_FEATURES = list(FEATURE_NUMERIC) + list(CATEGORICAL_COLS)
X_train, X_test, y_train, y_test = train_test_split(
    df_clean[ALL_FEATURES], df_clean[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE)
log(f"Training rows: {len(X_train):,} ({1 - TEST_SIZE:.0%}) | Test rows: {len(X_test):,} ({TEST_SIZE:.0%}) "
    f"| random_state={RANDOM_STATE}")
log("The same split is used for every model. Imputation, scaling, polynomial expansion and "
    "one-hot encoding all live inside scikit-learn Pipelines, so they are fitted on the training "
    "rows only (no data leakage). The test set is used once per model, for the final metrics.")



def make_linear_pipeline(numeric, categorical, scale=False):
    """Linear Regression with preprocessing fitted on training data only."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("num", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return Pipeline([("prep", ColumnTransformer(transformers, sparse_threshold=0)),
                     ("model", LinearRegression())])


def make_polynomial_pipeline(degree):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),  
        ("model", LinearRegression()),
    ])


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


RESULTS, PREDICTIONS, MODELS = [], {}, {}


def run_model(key, label, model_name, features_desc, degree, model, columns):
    """Fit on the training set, evaluate on the test set (+ 5-fold CV on the training set)."""
    Xtr, Xte = X_train[columns], X_test[columns]
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    train_pred = model.predict(Xtr)
    test_m = regression_metrics(y_test, pred)
    cv = cross_validate(clone(model), Xtr, y_train,
                        cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                        scoring=("neg_root_mean_squared_error", "r2"))
    RESULTS.append({
        "Key": key, "Label": label, "Model": model_name, "Features": features_desc, "Degree": degree,
        "N_Features": len(columns), **test_m,
        "Train_R2": r2_score(y_train, train_pred),
        "Train_RMSE": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "CV_RMSE": float(-cv["test_neg_root_mean_squared_error"].mean()),
        "CV_R2": float(cv["test_r2"].mean()),
    })
    PREDICTIONS[key], MODELS[key] = pred, model
    return test_m


section("8. LINEAR REGRESSION - BASELINE (Study Hours -> Exam Score)")
baseline_key = "LIN_A"
letter_a, cols_a, desc_a = FEATURE_SETS[0]
m = run_model(baseline_key, f"{letter_a}: {desc_a}", "Linear Regression", desc_a, 1,
              make_linear_pipeline(cols_a, []), cols_a)
baseline = MODELS[baseline_key]
intercept = float(baseline.named_steps["model"].intercept_)
slope = float(baseline.named_steps["model"].coef_[0])
sign = "+" if slope >= 0 else "-"
BASELINE_EQUATION = f"{TARGET} = {intercept:.3f} {sign} {abs(slope):.4f} x {HOURS}"
log(f"Regression equation: {BASELINE_EQUATION}")
log(f"  -> one more unit of {pretty(HOURS)} is associated with {slope:+.3f} exam-score points "
    "in this model (an association, not proof of cause).")
log(f"Test-set metrics: MAE={m['MAE']:.3f}  MSE={m['MSE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")
log("""
What the metrics mean:
  MAE  - the average size of the prediction error, in exam-score points (easy to interpret).
  MSE  - the average squared error; it punishes large mistakes much more than small ones.
  RMSE - the square root of MSE, back in score points; a 'typical' error that is sensitive to big misses.
  R2   - the share of the variation in exam scores the model explains (0 = no better than always
         predicting the mean score, 1 = perfect).""")


section("9. POLYNOMIAL REGRESSION (Study Hours only)")
for degree in POLY_DEGREES:
    m = run_model(f"POLY_{degree}", f"Poly deg {degree}: {pretty(HOURS)}", "Polynomial Regression",
                  desc_a, degree, make_polynomial_pipeline(degree), [HOURS])
    log(f"Degree {degree}: MAE={m['MAE']:.3f}  MSE={m['MSE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")


section("10. FEATURE COMBINATION EXPERIMENTS (Linear Regression)")
for letter, cols, desc in FEATURE_SETS[1:]:
    numeric = [c for c in cols if c in FEATURE_NUMERIC]
    categorical = [c for c in cols if c in CATEGORICAL_COLS]
    m = run_model(f"LIN_{letter}", f"{letter}: {desc}", "Linear Regression", desc, 1,
                  make_linear_pipeline(numeric, categorical), cols)
    log(f"Model {letter} [{desc}]: MAE={m['MAE']:.3f}  MSE={m['MSE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")


section("11. MODEL EVALUATION")
results = pd.DataFrame(RESULTS)
base = results.loc[results["Key"] == baseline_key].iloc[0]
results["RMSE_change_vs_baseline_pct"] = (results["RMSE"] / base["RMSE"] - 1) * 100
results["R2_change_vs_baseline"] = results["R2"] - base["R2"]
results["Train_Test_R2_gap"] = results["Train_R2"] - results["R2"]

best = results.loc[results["RMSE"].idxmin()]
BEST_KEY = best["Key"]
best_cv = results.loc[results["CV_RMSE"].idxmin()]
log("Test-set comparison (sorted by test RMSE, lower is better):")
show = results.sort_values("RMSE")[["Label", "MAE", "MSE", "RMSE", "R2", "Train_R2", "CV_RMSE"]]
log(show.round(4).to_string(index=False))
log(f"\nBest model on the TEST set: {best['Label']} (RMSE={best['RMSE']:.3f}, R2={best['R2']:.3f}). "
    f"Best on training-set cross-validation: {best_cv['Label']} "
    f"({'same model' if best_cv['Key'] == BEST_KEY else 'different model'}).")
gap = results["Train_Test_R2_gap"]
log(f"\nOverfitting check (training R2 minus test R2): smallest {gap.min():+.4f}, largest {gap.max():+.4f}.")
if gap.max() < 0.02:
    log("  -> No model fits the training data noticeably better than the test data, so there is no sign of "
        f"overfitting. In fact the test R2 is HIGHER than the training R2 for {int((gap < 0).sum())} of "
        f"{len(gap)} models (see 11.3 for the likely reason).")
else:
    worst = results.loc[gap.idxmax(), "Label"]
    log(f"  -> {worst} fits the training data clearly better than the test data: possible overfitting.")
poly_rows = results[results["Model"] == "Polynomial Regression"]
log("  Polynomial degrees vs. linear baseline (test RMSE change): "
    + ", ".join(f"deg {int(r.Degree)}: {r.RMSE_change_vs_baseline_pct:+.2f}%" for r in poly_rows.itertuples()))

log("\n--- 11.1 Coefficient analysis ---")
log("Numeric features are standardised (StandardScaler, fitted on training data) so their "
    "coefficients mean 'exam-score points per +1 standard deviation of the feature' and can be compared. "
    "Dummy (0/1) variables are left unscaled: their coefficient is the score difference "
    "versus the reference category. The target is never scaled, so everything is in score points.")


def coefficient_table(columns):
    numeric = [c for c in columns if c in FEATURE_NUMERIC]
    categorical = [c for c in columns if c in CATEGORICAL_COLS]
    scaled = make_linear_pipeline(numeric, categorical, scale=True).fit(X_train[columns], y_train)
    raw = make_linear_pipeline(numeric, categorical, scale=False).fit(X_train[columns], y_train)
    names = [n.split("__", 1)[1] for n in scaled.named_steps["prep"].get_feature_names_out()]
    table = pd.DataFrame({"Feature": names, "Coefficient": scaled.named_steps["model"].coef_})
    table["Raw_Coefficient"] = raw.named_steps["model"].coef_
    table["Type"] = ["numeric (per +1 SD)" if n in numeric else "category (vs reference)" for n in names]
    if categorical:
        onehot = scaled.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        reference = {c: cats[0] for c, cats in zip(categorical, onehot.categories_)}
        table["Reference"] = [next((f"{c} = {reference[c]}" for c in categorical if n.startswith(c + "_")), "")
                              for n in names]
    else:
        table["Reference"] = ""
    table["Abs"] = table["Coefficient"].abs()
    return table.sort_values("Abs", ascending=False).drop(columns="Abs").reset_index(drop=True)


numeric_all_key = next((f"LIN_{l}" for l, c, _ in FEATURE_SETS if set(c) == set(FEATURE_NUMERIC) and len(c) > 1), None)
full_key = next((f"LIN_{l}" for l, c, _ in FEATURE_SETS if set(c) == set(ALL_FEATURES) and len(c) > 1), None)
COEF_TABLES = {}
for key in (numeric_all_key, full_key):
    if key is None:
        continue
    cols = next(c for l, c, _ in FEATURE_SETS if f"LIN_{l}" == key)
    COEF_TABLES[key] = coefficient_table(cols)
    log(f"\nCoefficients of {results.loc[results['Key'] == key, 'Label'].iloc[0]}:")
    log(COEF_TABLES[key].round(4).to_string(index=False))
    pos = COEF_TABLES[key].query("Coefficient > 0")
    neg = COEF_TABLES[key].query("Coefficient < 0")
    log(f"  strongest positive: {pos.head(3)['Feature'].tolist()} | negative coefficients: "
        f"{neg['Feature'].tolist() if len(neg) else 'none'} | weakest: "
        f"{COEF_TABLES[key].tail(3)['Feature'].tolist()}")
log("\nInterpretation caution: a coefficient is the model's estimated association holding the OTHER "
    "INCLUDED variables constant. It does not prove causation, and omitted variables can shift it.")


def error_analysis(key):
    label = results.loc[results["Key"] == key, "Label"].iloc[0]
    pred = PREDICTIONS[key]
    resid = y_test.to_numpy() - pred
    mae, rmse = np.mean(np.abs(resid)), np.sqrt(np.mean(resid ** 2))
    log(f"\n--- 11.2 Error analysis: {label} ---")
    log(f"Mean error (bias) = {resid.mean():+.4f} | median = {np.median(resid):+.3f} | std = {resid.std():.3f} "
        f"| MAE = {mae:.3f} | RMSE = {rmse:.3f}")
    log(f"Residual range: {resid.min():+.2f} to {resid.max():+.2f} | skew = {stats.skew(resid):+.2f} | "
        f"excess kurtosis = {stats.kurtosis(resid):+.2f}")
    log(f"Share of predictions within +/-1 point: {np.mean(np.abs(resid) <= 1):.1%}, "
        f"+/-2: {np.mean(np.abs(resid) <= 2):.1%}, +/-3: {np.mean(np.abs(resid) <= 3):.1%}, "
        f"+/-5: {np.mean(np.abs(resid) <= 5):.1%}")
    log(f"Overpredicted (pred > actual): {np.mean(resid < 0):.1%} | underpredicted: {np.mean(resid > 0):.1%}")

    edges = np.unique(np.quantile(y_test, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    edges[0] -= 1
    labels = [f"{int(a) + 1}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])]
    tail = pd.DataFrame({"actual": y_test.to_numpy(), "pred": pred, "resid": resid})
    tail["Actual score range"] = pd.cut(tail["actual"], edges, labels=labels)
    by_range = tail.groupby("Actual score range", observed=True).agg(
        Students=("resid", "size"), Mean_error=("resid", "mean"), MAE=("resid", lambda r: np.mean(np.abs(r))),
        RMSE=("resid", lambda r: np.sqrt(np.mean(r ** 2)))).round(3)
    log("\nError by actual-score range (mean error > 0 means the model UNDERpredicts these students):")
    log(by_range.to_string())
    low_half = tail["pred"] <= np.median(tail["pred"])
    log(f"\nResidual std for lower-half vs upper-half predictions: {tail.loc[low_half, 'resid'].std():.3f} vs "
        f"{tail.loc[~low_half, 'resid'].std():.3f} (similar values = roughly constant error spread)")
    log("Note: grouping by ACTUAL score always shows some regression-to-the-mean (low scores look "
        "overpredicted, high scores underpredicted), even for a fair model; the residual-vs-predicted "
        "plot is the cleaner check for systematic bias.")
    return by_range, tail


by_range_best, tail_best = error_analysis(BEST_KEY)
if BEST_KEY != baseline_key:
    by_range_base, _ = error_analysis(baseline_key)

LARGE_ERROR = 5  


def error_decomposition():
    rows = []
    for r in results.itertuples():
        resid = y_test.to_numpy() - PREDICTIONS[r.Key]
        big = np.abs(resid) > LARGE_ERROR
        rows.append({
            "Label": r.Label, "Test_rows": len(resid), "Rows_with_error_gt_5": int(big.sum()),
            "Share_of_rows": big.mean(), "Share_of_total_squared_error": (resid[big] ** 2).sum() / (resid ** 2).sum(),
            "RMSE_all_rows": float(np.sqrt(np.mean(resid ** 2))),
            "RMSE_without_them": float(np.sqrt(np.mean(resid[~big] ** 2))) if (~big).any() else np.nan,
            "MAE_without_them": float(np.mean(np.abs(resid[~big]))) if (~big).any() else np.nan})
    return pd.DataFrame(rows)


decomposition = error_decomposition()
log(f"\n--- 11.3 Error decomposition: test rows with an error larger than {LARGE_ERROR} points ---")
log("(Diagnostic only - these rows are NOT removed from any reported metric.)")
log(decomposition.round(4).to_string(index=False))
best_cols = next((c for l, c, _ in FEATURE_SETS if f"LIN_{l}" == BEST_KEY), [HOURS])
train_resid = y_train.to_numpy() - MODELS[BEST_KEY].predict(X_train[best_cols])
n_big_train, n_big_test = int((np.abs(train_resid) > LARGE_ERROR).sum()), int(decomposition.loc[decomposition["Label"] == best["Label"], "Rows_with_error_gt_5"].iloc[0])
share_train, share_test = n_big_train / len(train_resid), n_big_test / len(y_test)
log(f"\n{best['Label']}: {n_big_train} of {len(train_resid):,} training rows ({share_train:.2%}) and "
    f"{n_big_test} of {len(y_test):,} test rows ({share_test:.2%}) have an error above {LARGE_ERROR} points.")
if share_test < share_train and best["R2"] > best["Train_R2"]:
    log(f"  -> This split put a smaller share of these hard-to-predict students in the test set, which is why "
        f"its test R2 ({best['R2']:.3f}) is higher than its training R2 ({best['Train_R2']:.3f}). "
        f"The 5-fold cross-validated R2 on the training data ({best['CV_R2']:.3f}) is the more cautious estimate "
        "of how the model would do on new students.")

base_resid = y_test.to_numpy() - PREDICTIONS[baseline_key]
resid_corr = X_test[FEATURE_NUMERIC].corrwith(pd.Series(base_resid, index=X_test.index)).drop(HOURS).round(3)
log("\nCorrelation of the BASELINE model's test residuals with the features it did not use "
    "(large values = missing information):")
log(resid_corr.sort_values(key=np.abs, ascending=False).to_string())
lin_rmse = results.loc[results["Key"] == baseline_key, "RMSE"].iloc[0]
for degree in POLY_DEGREES:
    row = results.loc[results["Key"] == f"POLY_{degree}"].iloc[0]
    log(f"Polynomial degree {degree} vs linear (hours only): RMSE {lin_rmse:.3f} -> {row['RMSE']:.3f} "
        f"({row['RMSE_change_vs_baseline_pct']:+.2f}%), R2 {base['R2']:.3f} -> {row['R2']:.3f}")


section("12. VISUALIZATIONS (model charts)")
rng = np.random.default_rng(RANDOM_STATE)
hours_grid = np.linspace(df_clean[HOURS].min(), df_clean[HOURS].max(), 300)
grid_frame = pd.DataFrame({HOURS: hours_grid})


def jitter(values, amount=0.25):
    return np.asarray(values) + rng.uniform(-amount, amount, len(values))


def plot_linear_regression():
    resid = y_test.to_numpy() - PREDICTIONS[baseline_key]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.scatter(jitter(X_train[HOURS]), y_train, s=9, alpha=0.15, color=GREY, label="Training data")
    ax.scatter(jitter(X_test[HOURS]), y_test, s=14, alpha=0.5, color=BLUE, label="Test data (actual)")
    ax.scatter(X_test[HOURS], PREDICTIONS[baseline_key], s=10, color=GREEN, alpha=0.6, label="Test data (predicted)")
    ax.plot(hours_grid, baseline.predict(grid_frame), color=RED, lw=3, label="Fitted regression line")
    ax.set(title="Linear Regression: Study Hours vs. Final Score\n" + BASELINE_EQUATION.replace(" x ", " × "),
           xlabel=pretty(HOURS), ylabel=pretty(TARGET))
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.scatter(jitter(X_test[HOURS]), resid, s=14, alpha=0.4, color=BLUE)
    per_hour = pd.DataFrame({"h": X_test[HOURS].to_numpy(), "r": resid}).groupby("h")["r"].agg(["mean", "count"])
    per_hour = per_hour[per_hour["count"] >= 5]
    ax.plot(per_hour.index, per_hour["mean"], "-o", color=RED, ms=4, label="Mean residual per hours value")
    ax.axhline(0, color="black", lw=1)
    ax.set(title="Residuals of the Linear Baseline (test set)", xlabel=pretty(HOURS),
           ylabel="Residual = actual - predicted")
    ax.legend()
    save_fig(fig, "linear_regression.png")


def plot_polynomial_regression():
    train_means = pd.DataFrame({"h": X_train[HOURS].to_numpy(), "y": y_train.to_numpy()}).groupby("h")["y"].agg(["mean", "count"])
    train_means = train_means[train_means["count"] >= 5]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(jitter(X_test[HOURS]), y_test, s=12, alpha=0.3, color=GREY, label="Test data")
    ax.plot(train_means.index, train_means["mean"], "o", color="black", ms=4, label="Training mean per hours value")
    styles = [(baseline_key, "Linear (degree 1)", RED, "-")] + \
             [(f"POLY_{d}", f"Polynomial degree {d}", c, ls) for d, c, ls in zip(POLY_DEGREES, (BLUE, GREEN, "#8172B2"), ("-", "--", "-."))]
    for key, name, color, ls in styles:
        rmse = results.loc[results["Key"] == key, "RMSE"].iloc[0]
        r2 = results.loc[results["Key"] == key, "R2"].iloc[0]
        ax.plot(hours_grid, MODELS[key].predict(grid_frame), color=color, ls=ls, lw=2.5,
                label=f"{name} (test RMSE {rmse:.2f}, R² {r2:.3f})")
    ax.set(title="Study Hours vs. Final Score: Linear vs. Polynomial Fits", xlabel=pretty(HOURS), ylabel=pretty(TARGET))
    ax.legend(loc="upper left")
    save_fig(fig, "polynomial_regression.png")


def plot_actual_vs_predicted():
    keys = [baseline_key] if BEST_KEY == baseline_key else [baseline_key, BEST_KEY]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.5 * len(keys), 6.5), squeeze=False)
    lo, hi = min(y_test.min(), min(PREDICTIONS[k].min() for k in keys)), max(y_test.max(), max(PREDICTIONS[k].max() for k in keys))
    for ax, key in zip(axes[0], keys):
        row = results.loc[results["Key"] == key].iloc[0]
        ax.scatter(jitter(y_test, 0.2), PREDICTIONS[key], s=14, alpha=0.4, color=BLUE)
        ax.plot([lo, hi], [lo, hi], color=RED, ls="--", lw=2, label="Perfect prediction")
        ax.set(title=f"{row['Label']}\nRMSE = {row['RMSE']:.2f}, R² = {row['R2']:.3f}",
               xlabel=f"Actual {pretty(TARGET)}", ylabel=f"Predicted {pretty(TARGET)}")
        ax.legend(loc="upper left")
    fig.suptitle("Actual vs. Predicted Exam Scores (test set)", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "actual_vs_predicted.png")


def plot_model_comparison():
    ordered = results.sort_values("RMSE", ascending=False)
    colors = [RED if k == BEST_KEY else BLUE for k in ordered["Key"]]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    axes[0].barh(ordered["Label"], ordered["RMSE"], color=colors)
    axes[0].set(title="Test RMSE (lower is better)", xlabel="RMSE (exam-score points)")
    axes[1].barh(ordered["Label"], ordered["R2"], color=colors)
    axes[1].set(title="Test R² (higher is better)", xlabel="R²")
    for ax, col, fmt in zip(axes, ("RMSE", "R2"), ("{:.2f}", "{:.3f}")):
        for y, v in enumerate(ordered[col]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9)
        ax.set_xlim(0, ordered[col].max() * 1.15)
    fig.suptitle("Model Performance Comparison (best model in red)", fontsize=15)
    fig.tight_layout()
    save_fig(fig, "model_comparison.png")


def plot_train_vs_test():
    ordered = results.sort_values("R2")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y + 0.2, ordered["Train_R2"], height=0.4, color=GREY, label="Training R²")
    ax.barh(y - 0.2, ordered["R2"], height=0.4, color=BLUE, label="Test R²")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["Label"])
    ax.set(title="Overfitting Check: Training vs. Test R²", xlabel="R²")
    ax.legend(loc="lower right")
    save_fig(fig, "train_vs_test_r2.png")


def plot_residuals():
    pred = PREDICTIONS[BEST_KEY]
    resid = y_test.to_numpy() - pred
    label = results.loc[results["Key"] == BEST_KEY, "Label"].iloc[0]
    far = np.abs(resid) > LARGE_ERROR
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]  
    ax.scatter(pred, resid, s=14, alpha=0.4, color=BLUE)
    ax.axhline(0, color="black", lw=1)
    ax.set(title=f"Residuals vs. Predicted (full range)\n{int(far.sum())} of {len(resid):,} students have an error > {LARGE_ERROR} points",
           xlabel="Predicted score", ylabel="Residual = actual - predicted")

    ax = axes[0, 1]  
    ax.scatter(pred, resid, s=14, alpha=0.4, color=BLUE)
    deciles = pd.qcut(pd.Series(pred), 10, duplicates="drop")
    med = pd.DataFrame({"p": pred, "r": resid}).groupby(deciles.to_numpy(), observed=True).agg(p=("p", "mean"), r=("r", "median"))
    ax.plot(med["p"], med["r"], "-o", color=RED, ms=4, label="Median residual per prediction decile")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylim(-LARGE_ERROR / 2.5, LARGE_ERROR / 2.5)
    ax.set(title="Residuals vs. Predicted (zoomed to typical students)", xlabel="Predicted score",
           ylabel="Residual = actual - predicted")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    bulk = resid[~far]
    sns.histplot(bulk, bins=40, kde=False, color=BLUE, stat="density", ax=ax)
    xs = np.linspace(bulk.min(), bulk.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, bulk.mean(), bulk.std()), color=RED, ls="--", label="Normal curve (same mean/SD)")
    ax.set(title=f"Distribution of Residuals (|error| <= {LARGE_ERROR} only)", xlabel="Residual", ylabel="Density")
    ax.text(0.98, 0.72, f"{int(far.sum())} residual(s) beyond ±{LARGE_ERROR}\nnot shown (largest {resid.max():+.1f})",
            transform=ax.transAxes, ha="right", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec=GREY))
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    colors = [GREEN if v >= 0 else RED for v in by_range_best["Mean_error"]]
    ax.bar(by_range_best.index.astype(str), by_range_best["Mean_error"], color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set(title="Mean Error by Actual Score Range\n(green = underpredicted, red = overpredicted)",
           xlabel="Actual score range", ylabel="Mean residual")
    fig.suptitle(f"Residual Analysis - {label}", fontsize=15)
    fig.tight_layout()
    save_fig(fig, "residual_plot.png")


def plot_coefficients():
    keys = [k for k in (numeric_all_key, full_key) if k in COEF_TABLES]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(8.5 * len(keys), 7), squeeze=False)
    for ax, key in zip(axes[0], keys):
        table = COEF_TABLES[key].head(12).iloc[::-1]
        ax.barh(table["Feature"], table["Coefficient"], color=[GREEN if v >= 0 else RED for v in table["Coefficient"]])
        ax.axvline(0, color="black", lw=1)
        ax.set(title=results.loc[results["Key"] == key, "Label"].iloc[0], xlabel="Effect on predicted score (points)")
    fig.suptitle("Largest Coefficients (numeric: per +1 SD; categories: vs. reference group)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "coefficients.png")


plot_linear_regression()
plot_polynomial_regression()
plot_actual_vs_predicted()
plot_model_comparison()
plot_train_vs_test()
plot_residuals()
plot_coefficients()


section("13. MODEL COMPARISON")
table_columns = ["Model", "Features", "Degree", "MAE", "MSE", "RMSE", "R2"]
comparison = results[table_columns + ["Train_R2", "CV_RMSE", "CV_R2", "RMSE_change_vs_baseline_pct",
                                      "R2_change_vs_baseline"]].copy()
comparison.insert(0, "Label", results["Label"])
log("Required comparison table (all values are TEST-set metrics):")
log(results[table_columns].round(4).to_string(index=False))
log("\nExtra columns: Train_R2 (fit on training data), CV_RMSE / CV_R2 (5-fold cross-validation on the "
    "training data only).")
log(comparison[["Label", "Train_R2", "R2", "CV_R2", "CV_RMSE", "RMSE", "RMSE_change_vs_baseline_pct"]].round(4)
    .to_string(index=False))


section("14. EXPORT RESULTS")
comparison.round(6).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
comparison[comparison["Model"] == "Linear Regression"].round(6).to_csv(
    REPORT_DIR / "feature_combination_results.csv", index=False)
for key, table in COEF_TABLES.items():
    table.round(6).to_csv(REPORT_DIR / f"coefficients_{key.lower()}.csv", index=False)
by_range_best.to_csv(REPORT_DIR / "error_by_score_range.csv")
decomposition.round(6).to_csv(REPORT_DIR / "error_decomposition.csv", index=False)
pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted_baseline": PREDICTIONS[baseline_key].round(3),
              "Predicted_best": PREDICTIONS[BEST_KEY].round(3),
              "Residual_best": (y_test.to_numpy() - PREDICTIONS[BEST_KEY]).round(3)},
             index=y_test.index).to_csv(REPORT_DIR / "test_predictions.csv", index_label="row_index")
log("saved reports/model_comparison.csv, feature_combination_results.csv, coefficients_*.csv, "
    "error_by_score_range.csv, error_decomposition.csv, test_predictions.csv")

log(f"\nFINAL: best model on the test set = {best['Label']} | RMSE {best['RMSE']:.3f}, MAE {best['MAE']:.3f}, "
    f"R2 {best['R2']:.3f}. Baseline (hours only): RMSE {base['RMSE']:.3f}, R2 {base['R2']:.3f}.")
(REPORT_DIR / "analysis_summary.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
print("\nDone. All outputs are in data/, charts/ and reports/.")
