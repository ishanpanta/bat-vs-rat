import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ====================================================
# 0. Setup & Load the Datasets
# ====================================================


def load_and_clean_data(path1, path2):
    """
    Load cleaned bat (df1) and rat (df2) datasets.
    - Parse datetime columns with dayfirst=True
    - Convert month/season to categorical
    """

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # --- Parse datetimes ---
    datetime_cols_df1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
    for col in datetime_cols_df1:
        if col in df1.columns:
            df1[col] = pd.to_datetime(df1[col], errors="coerce", dayfirst=True)

    if "time" in df2.columns:
        df2["time"] = pd.to_datetime(df2["time"], errors="coerce", dayfirst=True)

    # --- Convert categorical ---
    if "month" in df1.columns:
        df1["month"] = df1["month"].astype(int).astype("category")
    if "season" in df1.columns:
        df1["season"] = df1["season"].astype("category")
    if "month" in df2.columns:
        df2["month"] = df2["month"].astype(int).astype("category")

    return df1, df2


# ====================================================
# 1. Descriptive Analysis
# ====================================================

def descriptive_stats(df1, df2):
    return {
        "dataset1_shape": df1.shape,
        "dataset2_shape": df2.shape,
        "dataset1_desc": df1.describe(include="all"),
        "dataset2_desc": df2.describe(include="all"),
    }


# ====================================================
# 2. Aggregation & Merging
# ====================================================

def aggregate_dataset1(df1):
    agg = (
        df1.groupby(["month", "hours_after_sunset"], observed=True)
        .agg({
            "bat_landing_to_food": "mean",
            "risk": "mean",
            "reward": "mean",
            "season": lambda x: x.mode()[0] if not x.mode().empty else None
        })
        .reset_index()
    )
    agg = agg.rename(columns={
        "bat_landing_to_food": "mean_bat_landing_to_food",
        "risk": "mean_risk",
        "reward": "mean_reward"
    })
    return agg

def merge_datasets(df1, df2):
    df1_agg = aggregate_dataset1(df1)
    merged = pd.merge(df2, df1_agg, on=["month", "hours_after_sunset"], how="inner")
    return merged


# ====================================================
# 3. Correlation Analysis
# ====================================================

def safe_corr(x, y):
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return x.corr(y)


def correlation_analysis(merged):
    return {
        "corr_rat_vs_delay": safe_corr(merged["rat_arrival_number"], merged["mean_bat_landing_to_food"]),
        "corr_rat_vs_risk": safe_corr(merged["rat_arrival_number"], merged["mean_risk"]),
        "corr_rat_minutes_vs_delay": safe_corr(merged["rat_minutes"], merged["mean_bat_landing_to_food"]),
        "corr_rat_minutes_vs_risk": safe_corr(merged["rat_minutes"], merged["mean_risk"]),
    }


# ====================================================
# 4. Group Comparison
# ====================================================

def group_comparison(merged):
    m = merged.dropna(subset=["rat_arrival_number", "mean_bat_landing_to_food", "mean_risk"])
    group0 = m[m["rat_arrival_number"] == 0]
    group1 = m[m["rat_arrival_number"] > 0]

    results = {
        "group0_mean_delay": group0["mean_bat_landing_to_food"].mean(),
        "group1_mean_delay": group1["mean_bat_landing_to_food"].mean(),
        "group0_mean_risk": group0["mean_risk"].mean(),
        "group1_mean_risk": group1["mean_risk"].mean(),
    }

    if len(group0) > 0 and len(group1) > 0:
        u1 = mannwhitneyu(group0["mean_bat_landing_to_food"], group1["mean_bat_landing_to_food"])
        u2 = mannwhitneyu(group0["mean_risk"], group1["mean_risk"])
        results["mannwhitney_delay_p"] = u1.pvalue
        results["mannwhitney_risk_p"] = u2.pvalue

    return results


# ====================================================
# 5. Regression Analysis
# ====================================================

def logistic_regression(df1, df2):
    import statsmodels.api as sm

    merged = merge_datasets(df1, df2)
    if merged['mean_risk'].nunique() > 1:
        try:
            X = merged[['rat_arrival_number', 'rat_minutes']].fillna(0)
            y = (merged['mean_risk'] > 0.5).astype(int)
            model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
            return {"dataset": "merged", "summary": model.summary().as_text()}
        except Exception as e:
            return {"dataset": "merged", "error": str(e)}

    if df1['risk'].nunique() > 1:
        try:
            X = df1[['seconds_after_rat_arrival']].fillna(0)
            y = df1['risk']
            model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
            return {"dataset": "raw", "summary": model.summary().as_text()}
        except Exception as e:
            return {"dataset": "raw", "error": str(e)}

    return {"dataset": None, "message": "Not enough variation in risk for regression."}


# ====================================================
# 6. Visualization (Investigation A & B)
# ====================================================

def plot_relationships(merged):
    plt.figure()
    plt.scatter(merged["rat_arrival_number"], merged["mean_bat_landing_to_food"])
    plt.xlabel("Rat arrivals")
    plt.ylabel("Bat approach delay (s)")
    plt.title("Rat arrivals vs Bat approach delay")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.scatter(merged["rat_arrival_number"], merged["mean_risk"])
    plt.xlabel("Rat arrivals")
    plt.ylabel("Proportion risk-taking")
    plt.title("Rat arrivals vs Bat risk-taking")
    plt.grid(True)
    plt.show()


def plot_by_season(merged):
    if "season" not in merged.columns:
        print("No season column available in merged data.")
        return
    (
        merged.groupby("season", observed=True)[["mean_bat_landing_to_food", "mean_risk"]]
        .mean()
        .plot(kind="bar")
    )
    plt.title("Seasonal differences in bat behaviour")
    plt.ylabel("Mean values")
    plt.grid(True)
    plt.show()


def exploratory_plots(df1, df2):
    # Risk vs Reward
    sns.countplot(x="risk", hue="reward", data=df1)
    plt.title("Risk-taking vs Reward Outcome")
    plt.xticks([0, 1], ["Avoidance", "Risk-taking"])
    plt.show()

    # Risk-taking %
    risk_rate = df1["risk"].value_counts(normalize=True) * 100
    print("Risk-taking behavior rate:\n", risk_rate)

    # Approach Time vs Risk
    sns.boxplot(x="risk", y="bat_landing_to_food", data=df1)
    plt.title("Approach Time vs Risk-taking")
    plt.xticks([0, 1], ["Avoidance", "Risk-taking"])
    plt.show()

    # Vigilance by Time since Rat arrival
    sns.scatterplot(x="seconds_after_rat_arrival", y="bat_landing_to_food", hue="risk", data=df1)
    plt.title("Time After Rat Arrival vs Bat Landing Response")
    plt.show()

    # Seasonal Effect
    sns.countplot(x="season", hue="risk", data=df1)
    plt.title("Seasonal Risk Behavior")
    plt.show()

    sns.countplot(x="season", hue="reward", data=df1)
    plt.title("Seasonal Reward Behavior")
    plt.show()

    sns.boxplot(x="month", y="bat_landing_number", data=df2)
    plt.title("Monthly Bat Landings")
    plt.xticks(rotation=45)
    plt.show()


# ====================================================
# 7. Summary Stats
# ====================================================

def extra_summary(df1):
    corr = df1.corr(numeric_only=True)
    reward_by_season = df1.groupby("season")["reward"].mean()
    return {"correlation_matrix": corr, "reward_by_season": reward_by_season}


# ====================================================
# 8. Master Function
# ====================================================

def run_investigation_a(df1, df2):
    merged = merge_datasets(df1, df2)

    results = {
        "descriptive": descriptive_stats(df1, df2),
        "correlation": correlation_analysis(merged),
        "group_comparison": group_comparison(merged),
        "regression": logistic_regression(df1, df2),
        "extra_summary": extra_summary(df1),
    }

    # Plots
    plot_relationships(merged)
    plot_by_season(merged)
    exploratory_plots(df1, df2)

    return results
