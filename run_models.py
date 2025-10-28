from libraries import *
from models import Training_Model
from functions import CNN_Params, MLP_Params
from prints import backtest_model
from models import  _load_latest ,register_models
from data_processing import all_indicators, all_normalization_indicators, all_normalization_price, dataset_split, clean_data, process_dataset, target


def Base():
    """
    Main training and backtesting pipeline.

    This function performs the complete workflow for model training and evaluation:
    1. Loads and splits historical financial data into training, testing, and validation sets.
    2. Processes datasets to generate indicators, normalization, and targets.
    3. Builds and prepares feature matrices for both MLP and CNN architectures.
    4. Trains both models using predefined hyperparameters.
    5. Registers and reloads trained models for evaluation.
    6. Runs backtesting on each dataset split and visualizes portfolio performance.

    The workflow uses Apple (AAPL) 15-year data as a base example.

    Steps
    -----
    - Data preparation
    - Feature and target extraction
    - Model training (MLP & CNN)
    - Model registration and reloading
    - Backtesting and visualization

    Returns
    -------
    None
        The function runs the entire pipeline and prints training and backtest results.
    """
    data = clean_data("AAPL", "15y")
    train, test, val = dataset_split(data)

    train_data, train_ind, train_price = process_dataset(train, alpha=0.01)

    test_ind = all_normalization_indicators(all_indicators(test))
    test_price = all_normalization_price(all_indicators(test))
    val_ind = all_normalization_indicators(all_indicators(val))
    val_price = all_normalization_price(all_indicators(val))

    x_train, y_train = target(train_price)
    feature_cols = x_train.columns

    x_test = test_price.reindex(columns=feature_cols).fillna(0)
    x_val = val_price.reindex(columns=feature_cols).fillna(0)

    x_train = np.asarray(x_train, dtype="float32")
    x_test = np.asarray(x_test, dtype="float32")
    x_val = np.asarray(x_val, dtype="float32")

    datasets_mlp = {
        "train": (train_data.copy(), x_train),
        "test": (test.copy().iloc[-len(x_test):], x_test),
        "val": (val.copy().iloc[-len(x_val):], x_val),
    }

    x_train_c = x_train[..., None]
    x_test_c = x_test[..., None]
    x_val_c = x_val[..., None]

    datasets_cnn = {
        "train": (train_data.copy(), x_train_c),
        "test": (test.copy().iloc[-len(x_test_c):], x_test_c),
        "val": (val.copy().iloc[-len(x_val_c):], x_val_c),
    }

    mlp_params = MLP_Params()
    cnn_params = CNN_Params()

    print("\n--- TRAINING MLP ---")
    model_MLP = Training_Model.training_MLP(
        x_train, y_train,
        x_test, None,
        x_val, None,
        [mlp_params]
    )

    print("\n--- TRAINING CNN ---")
    model_CNN = Training_Model.training_CNN(
        x_train_c, y_train,
        x_test_c, None,
        x_val_c, None,
        [cnn_params]
    )

    models_to_register = {
        "MLP_Model": model_MLP,
        "CNN_Model": model_CNN,
    }
    register_models(models_to_register)

    mlp_loaded = _load_latest("MLP_Model_003")
    cnn_loaded = _load_latest("CNN_Model_003")

    joblib.dump(feature_cols, "feature_cols.pkl")

    backtest_model(datasets_mlp, mlp_loaded, "MLP")
    backtest_model(datasets_cnn, cnn_loaded, "CNN")


if __name__ == "__main__":
    Base()