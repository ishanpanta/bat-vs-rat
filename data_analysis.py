# analysis_functions.py

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# 1. Descriptive Analysis

def descriptive_stats(df1, df2):
    return {
        "dataset1_shape": df1.shape,
        "dataset2_shape": df2.shape,
        "dataset1_desc": df1.describe(include="all"),
        "dataset2_desc": df2.describe(include="all"),
    }


# 2. Aggregation & Merging

def aggregate_dataset1(df1):
    """
    Aggregate dataset1 by month and hours_after_sunset.
    Keep season information (mode per group).
    """
    agg = df1.groupby(["month", "hours_after_sunset"]).agg({
        "bat_landing_to_food": "mean",
        "risk": "mean",
        "reward": "mean",
        "season": lambda x: x.mode()[0] if not x.mode().empty else None
    }).reset_index()

    agg = agg.rename(columns={
        "bat_landing_to_food": "mean_bat_landing_to_food",
        "risk": "mean_risk",
        "reward": "mean_reward"
    })
    return agg


def merge_datasets(df1, df2):
    df1_agg = aggregate_dataset1(df1)
    merged = pd.merge(df2, df1_agg, on=["month", "hours_after_sunset"],
                      how="inner")
    return merged


# 3. Correlation Analysis

def correlation_analysis(merged):
    return {
        "corr_rat_vs_delay":
        merged["rat_arrival_number"].corr(merged["mean_bat_landing_to_food"]),
        "corr_rat_vs_risk":
            merged["rat_arrival_number"].corr(merged["mean_risk"]),
        "corr_rat_minutes_vs_delay":
            merged["rat_minutes"].corr(merged["mean_bat_landing_to_food"]),
        "corr_rat_minutes_vs_risk":
            merged["rat_minutes"].corr(merged["mean_risk"]),
    }


# 4. Group Comparison

def group_comparison(merged):
    m = merged.dropna(subset=["rat_arrival_number", "mean_bat_landing_to_food",
                              "mean_risk"])
    group0 = m[m["rat_arrival_number"] == 0]
    group1 = m[m["rat_arrival_number"] > 0]

    results = {
        "group0_mean_delay": group0["mean_bat_landing_to_food"].mean(),
        "group1_mean_delay": group1["mean_bat_landing_to_food"].mean(),
        "group0_mean_risk": group0["mean_risk"].mean(),
        "group1_mean_risk": group1["mean_risk"].mean(),
    }

    if len(group0) > 0 and len(group1) > 0:
        u1 = mannwhitneyu(group0["mean_bat_landing_to_food"], group1
                          ["mean_bat_landing_to_food"])
        u2 = mannwhitneyu(group0["mean_risk"], group1["mean_risk"])
        results["mannwhitney_delay_p"] = u1.pvalue
        results["mannwhitney_risk_p"] = u2.pvalue

    return results


# 5. Regression Analysis

def logistic_regression(df1, df2):
    """
    Logistic regression: does rat activity predict bat risk-taking?
    1. Try merged dataset (df1 + df2).
    2. If not enough variation, fall back to raw dataset1.
    3. If still no variation, return message.
    """
    import statsmodels.api as sm

    # --- Step 1: Try merged dataset ---
    merged = merge_datasets(df1, df2)
    if merged['mean_risk'].nunique() > 1:
        try:
            X = merged[['rat_arrival_number', 'rat_minutes']].fillna(0)
            y = (merged['mean_risk'] > 0.5).astype(int)
            model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
            return {"dataset": "merged", "summary": model.summary().as_text()}
        except Exception as e:
            return {"dataset": "merged", "error": str(e)}

    # --- Step 2: Fall back to raw dataset1 ---
    if df1['risk'].nunique() > 1:
        try:
            X = df1[['seconds_after_rat_arrival']].fillna(0)
            y = df1['risk']
            model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
            return {"dataset": "raw", "summary": model.summary().as_text()}
        except Exception as e:
            return {"dataset": "raw", "error": str(e)}

    # --- Step 3: No variation in risk at all ---
    return {"dataset": None, "message": "Not enough variation in risk for"
            "regression."}


# 6. Visualization


def plot_relationships(merged):
    plt.figure()
    plt.scatter(merged
                ["rat_arrival_number"], merged["mean_bat_landing_to_food"])
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
    merged.groupby("season")[["mean_bat_landing_to_food", "mean_risk"]
                             ].mean().plot(kind="bar")
    plt.title("Seasonal differences in bat behaviour")
    plt.ylabel("Mean values")
    plt.grid(True)
    plt.show()


# 7. Master Function

def run_investigation_a(df1, df2):
    """
    Run all analysis for Investigation A and return results.
    """
    merged = merge_datasets(df1, df2)

    results = {
        "descriptive": descriptive_stats(df1, df2),
        "correlation": correlation_analysis(merged),
        "group_comparison": group_comparison(merged),
        "regression": logistic_regression(df1, df2),
    }

    plot_relationships(merged)
    plot_by_season(merged)

    return results
