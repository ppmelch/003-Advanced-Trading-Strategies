
from libraries import *
from prints import backtest_model
from models import  model_name_version
from data_processing import all_indicators, all_normalization_indicators, dataset_split, clean_data

def main():
    """
    Evaluate pre-trained trading models on test and validation datasets.

    This function loads previously trained models and their corresponding
    feature set, reconstructs normalized test and validation datasets,
    and runs backtesting to assess portfolio performance.

    The workflow assumes models were already trained and registered via `Base()`
    and that `feature_cols.pkl` (feature order) exists in the working directory.

    Steps
    -----
    1. Load and split 15-year AAPL data into train, test, and validation sets.
    2. Normalize indicators for test and validation splits.
    3. Align features with the original training feature set.
    4. Load pre-trained MLP and CNN models by name and version.
    5. Perform backtesting for both models and display performance metrics.

    Returns
    -------
    None
        Executes evaluation and prints portfolio metrics and plots for both models.
    """
    print("🚀 Loading data for evaluation...\n")
    data = clean_data("AAPL", "15y")
    train, test, val = dataset_split(data)

    test_ind = all_normalization_indicators(all_indicators(test))
    val_ind = all_normalization_indicators(all_indicators(val))

    feature_cols = joblib.load("feature_cols.pkl")
    print(f"✅ Loaded training features: {len(feature_cols)} columns\n")

    test_ind = test_ind.reindex(columns=feature_cols).fillna(0)
    val_ind = val_ind.reindex(columns=feature_cols).fillna(0)

    x_test = np.asarray(test_ind, dtype="float32")
    x_val = np.asarray(val_ind, dtype="float32")

    datasets = {
        "test": (test.copy().iloc[-len(x_test):], x_test),
        "val": (val.copy().iloc[-len(x_val):], x_val),
    }

    model_MLP = model_name_version("MLP_Model_003", "10")
    model_CNN = model_name_version("CNN_Model_003", "10")

    backtest_model(datasets, model_MLP, "MLP")
    backtest_model(datasets, model_CNN, "CNN")


if __name__ == "__main__":
    main()
