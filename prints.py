from libraries import *
from metrics import metrics
from functions import Config
from backtesting import backtest
from visualization import plot_single_split, plot_test_and_validation


def results(
    cash: float,
    port_value,
    win_rate: float,
    buy: int,
    sell: int,
    hold: int,
    total_trades: int
) -> None:
    """
    Print backtest results and computed performance metrics.

    Parameters
    ----------
    cash : float
        Final cash value.
    port_value : pd.Series
        Time series of portfolio value for the evaluated period.
    win_rate : float
        Proportion of winning trades in [0, 1].
    buy : int
        Number of long entries executed.
    sell : int
        Number of short entries executed.
    hold : int
        Number of steps where no new position was opened.
    total_trades : int
        Total number of closed trades.

    Returns
    -------
    None
        Prints summary lines and calculated metrics to stdout.
    """
    final_value = float(port_value.iloc[-1])
    profit = final_value - Config.initial_capital
    profit_pct = (profit / Config.initial_capital) * 100

    print("="*60)
    print(f"Capital final:         ${cash:,.2f}")
    print(f"Valor del portafolio:  ${final_value:,.2f}")
    print(f"Ganancia total:        ${profit:,.2f} ({profit_pct:.2f}%)")
    print(f"Win Rate:              {win_rate*100:.2f}%")
    print(
        f"Operaciones -> Buy: {buy}, Sell: {sell}, Hold: {hold}, Total: {total_trades}")

    try:
        m = metrics(port_value)
        print("\n========================== MÉTRICAS ==========================")
        for k, v in m.items():
            print(f"{k:<20}: {v:.4f}" if isinstance(
                v, float) else f"{k:<20}: {v}")
        print("===============================================================\n")
    except Exception as e:
        print(f"\n⚠️ No se pudieron calcular métricas: {e}")


def backtest_model(datasets: dict, model, model_name: str) -> dict:
    """
    Run sequential backtests for 'train' → 'test' → 'val' for a given model.

    For each available split in `datasets`, it:
    1) Aligns features and predicts class probabilities with `model.predict`.
    2) Converts predictions to class labels, then to signals {-1, 0, 1} (shifted by 1 period).
    3) Executes the portfolio backtest.
    4) Prints results and plots individual curves for each split.
    5) Additionally plots a combined Test+Validation figure when both are present.
       The Test backtest is forced to start at 1,000,000; Validation starts
       where Test ended.

    Parameters
    ----------
    datasets : dict
        Mapping from split name to a tuple ``(dataframe, features)``.
        Expected keys include any of {'train', 'test', 'val'}.
        - dataframe : pd.DataFrame
            OHLCV data aligned to the features' index.
        - features : np.ndarray
            Model-ready feature matrix for the split.
    model : Any
        Trained model exposing a `.predict(x, verbose=0)` method that returns
        class probabilities for three classes (long/hold/short).
    model_name : str
        Name used for logging and plot titles.

    Returns
    -------
    dict
        Dictionary with keys {'train', 'test', 'val'} where present, each
        value being a pd.Series of portfolio value over time for that split.
    """
    order = ["train", "test", "val"]
    order = [k for k in order if k in (
        name.lower() for name in datasets.keys())]

    name_map = {name.lower(): name for name in datasets.keys()}

    pv = {"train": None, "test": None, "val": None}

    base_cash = Config.initial_capital

    print(
        f"\n================== RESULTS FOR MODEL: {model_name} ==================")

    if "train" in order:
        dataset_name = name_map["train"]
        data, x_data = datasets[dataset_name]

        if len(data) != len(x_data):
            data = data.iloc[-len(x_data):].copy()

        y_pred = model.predict(x_data, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        signals = pd.Series(y_pred_classes - 1,
                            index=data.index).shift(1).fillna(0).astype(int)

        port_series, cash, win_rate, buy, sell, hold, total_trades, *_ = backtest(
            data,
            signals=signals,
            cash=base_cash
        )
        pv["train"] = port_series
        print("\n--- TRAIN RESULTS ---")
        results(cash, port_series, win_rate, buy, sell, hold, total_trades)
        plot_single_split(port_series, f"{model_name} - Train Portfolio")

    if "test" in order:
        dataset_name = name_map["test"]
        data, x_data = datasets[dataset_name]

        if len(data) != len(x_data):
            data = data.iloc[-len(x_data):].copy()

        y_pred = model.predict(x_data, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        signals = pd.Series(y_pred_classes - 1,
                            index=data.index).shift(1).fillna(0).astype(int)

        port_series, cash, win_rate, buy, sell, hold, total_trades, *_ = backtest(
            data,
            signals=signals,
            cash=1_000_000.0
        )
        pv["test"] = port_series
        print("\n--- TEST RESULTS ---")
        results(cash, port_series, win_rate, buy, sell, hold, total_trades)
        plot_single_split(port_series, f"{model_name} - Test Portfolio")

    if "val" in order and pv["test"] is not None:
        dataset_name = name_map["val"]
        data, x_data = datasets[dataset_name]

        if len(data) != len(x_data):
            data = data.iloc[-len(x_data):].copy()

        y_pred = model.predict(x_data, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        signals = pd.Series(y_pred_classes - 1,
                            index=data.index).shift(1).fillna(0).astype(int)

        start_cash_val = float(pv["test"].iloc[-1])

        port_series, cash, win_rate, buy, sell, hold, total_trades, *_ = backtest(
            data,
            signals=signals,
            cash=start_cash_val
        )
        pv["val"] = port_series
        print("\n--- VAL RESULTS ---")
        results(cash, port_series, win_rate, buy, sell, hold, total_trades)
        plot_single_split(port_series, f"{model_name} - Validation Portfolio")

        if pv["test"] is not None:
            plot_test_and_validation(
                pv["test"], pv["val"], f"{model_name} - Test + Validation")

    return pv
