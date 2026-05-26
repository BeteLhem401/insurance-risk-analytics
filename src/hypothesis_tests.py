"""
hypothesis_tests.py

Reusable statistical hypothesis testing utilities
for insurance risk analytics.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency


def create_insurance_metrics(df):
    """
    Create LossRatio, Margin, and HasClaim features.
    """

    df = df.copy()

    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"]
    df["LossRatio"] = df["LossRatio"].replace([np.inf, -np.inf], np.nan)

    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    df["HasClaim"] = np.where(df["TotalClaims"] > 0, 1, 0)

    return df


def province_risk_test(df, province_col="Province"):
    """
    Compare loss ratios between top two provinces using t-test.
    """

    top_provinces = df[province_col].value_counts().index[:2]

    province_1 = top_provinces[0]
    province_2 = top_provinces[1]

    group1 = df[df[province_col] == province_1]["LossRatio"].dropna()
    group2 = df[df[province_col] == province_2]["LossRatio"].dropna()

    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

    decision = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"

    return {
        "Province_1": province_1,
        "Province_2": province_2,
        "T_Statistic": round(t_stat, 4),
        "P_Value": round(p_value, 6),
        "Decision": decision
    }


def postal_claim_frequency_test(df, postal_col="PostalCode"):
    """
    Chi-square test for claim frequency across postal codes.
    """

    contingency_table = pd.crosstab(
        df[postal_col],
        df["HasClaim"]
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    decision = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"

    return {
        "Chi2_Statistic": round(chi2, 4),
        "P_Value": round(p_value, 6),
        "Degrees_of_Freedom": dof,
        "Decision": decision
    }


def postal_margin_test(df, postal_col="PostalCode"):
    """
    T-test for margin difference between top two postal codes.
    """

    top_postal = df[postal_col].value_counts().index[:2]

    zip1 = top_postal[0]
    zip2 = top_postal[1]

    margin1 = df[df[postal_col] == zip1]["Margin"].dropna()
    margin2 = df[df[postal_col] == zip2]["Margin"].dropna()

    t_stat, p_value = ttest_ind(margin1, margin2, equal_var=False)

    decision = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"

    return {
        "PostalCode_1": zip1,
        "PostalCode_2": zip2,
        "T_Statistic": round(t_stat, 4),
        "P_Value": round(p_value, 6),
        "Decision": decision
    }


def gender_risk_test(df, gender_col="Gender"):
    """
    T-test for gender-based risk difference.
    """

    male = df[df[gender_col] == "Male"]["LossRatio"].dropna()
    female = df[df[gender_col] == "Female"]["LossRatio"].dropna()

    t_stat, p_value = ttest_ind(male, female, equal_var=False)

    decision = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"

    return {
        "T_Statistic": round(t_stat, 4),
        "P_Value": round(p_value, 6),
        "Decision": decision
    }


def create_summary_table(
    province_result,
    postal_frequency_result,
    postal_margin_result,
    gender_result
):
    """
    Create final summary table for all hypothesis tests.
    """

    summary = pd.DataFrame({
        "Hypothesis": [
            "Province Risk Difference",
            "Postal Code Claim Frequency",
            "Postal Code Margin Difference",
            "Gender Risk Difference"
        ],
        "P-Value": [
            province_result["P_Value"],
            postal_frequency_result["P_Value"],
            postal_margin_result["P_Value"],
            gender_result["P_Value"]
        ],
        "Decision": [
            province_result["Decision"],
            postal_frequency_result["Decision"],
            postal_margin_result["Decision"],
            gender_result["Decision"]
        ]
    })

    return summary


if __name__ == "__main__":

    print("hypothesis_tests.py loaded successfully.")

