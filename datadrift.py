from libraries import *
from scipy.stats import ks_2samp


def ks_drift_test(reference_df: pd.DataFrame, new_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform the Kolmogorov–Smirnov (KS) test to detect data drift 
    between numeric features of two datasets.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference dataset (e.g., training data).
    new_df : pd.DataFrame
        New dataset to compare against the reference (e.g., test or validation).
    alpha : float, default=0.05
        Significance level. If the p-value is below this threshold, drift is detected.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
            - 'feature' : feature name
            - 'p_value' : p-value from the KS test
            - 'drift_detected' : boolean indicating if drift was detected
        The DataFrame is sorted by ascending p-value.
    """
    results = []
    for feature in reference_df.columns:
        if pd.api.types.is_numeric_dtype(reference_df[feature]):
            p_value = ks_2samp(
                reference_df[feature].dropna(), new_df[feature].dropna()).pvalue
            drift_detected = p_value < alpha
            results.append({
                "feature": feature,
                "p_value": p_value,
                "drift_detected": drift_detected
            })
    return pd.DataFrame(results).sort_values("p_value")


def summarize_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Generate a summary dictionary of KS drift test results for each feature.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference dataset (e.g., training data).
    new_df : pd.DataFrame
        New dataset to compare against (e.g., test or validation).
    alpha : float, default=0.05
        Significance level for drift detection.

    Returns
    -------
    dict
        A dictionary where each key is a feature name and each value 
        contains the KS test result (p-value and drift flag).
    """
    drift_summary = {}
    for feature in reference_df.columns:
        result = ks_drift_test(
            reference_df[[feature]], new_df[[feature]], alpha)
        drift_summary[feature] = result
    return drift_summary


def compute_pvalues(reference_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Compute the Kolmogorov–Smirnov p-values for each feature 
    between a reference and new dataset.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline dataset (e.g., training data).
    new_df : pd.DataFrame
        Dataset to compare against (e.g., production or validation data).

    Returns
    -------
    dict
        A dictionary mapping each feature name to its KS test p-value.
    """
    p_values = {}
    for feature in reference_df.columns:
        _, p_value = ks_2samp(
            reference_df[feature].dropna(), new_df[feature].dropna())
        p_values[feature] = p_value
    return p_values
