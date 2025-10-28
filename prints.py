from functions import Config
from libraries import *
from metrics import metrics
from backtesting import backtest
from visualization import plot_test_validation

def show_results(cash: float, port_value: pd.Series, win_rate: float,
                 buy: int, sell: int, hold: int, total_trades: int) -> None:
    """
    Imprime resultados y métricas del backtest.
    """
    final_value = float(port_value.iloc[-1])
    profit = final_value - Config.initial_capital
    profit_pct = (profit / Config.initial_capital) * 100

    print(f"Capital final: ${cash:,.2f}")
    print(f"Valor del portafolio: ${final_value:,.2f}")
    print(f"Ganancia total: ${profit:,.2f} ({profit_pct:.2f}%)")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Operaciones -> Buy: {buy}, Sell: {sell}, Hold: {hold}, Total: {total_trades}")


def results(datasets: dict, model) -> None:
    print(f"\n================== RESULTS FOR MODEL: {model.name.upper()} ==================\n")

    # 🔹 Inicializa el capital antes del primer dataset
    cash = Config.initial_capital  

    for dataset_name, (data, x_data) in datasets.items():
        print(f"--- {dataset_name.upper()} RESULTS ---")

        # predicciones
        y_pred = model.predict(x_data, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        data["final_signal"] = y_pred_classes - 1

        # backtest con cash encadenado
        port_series, cash, win_rate, buy, sell, hold, total_trades, _, _ = backtest(
            data, cash=cash
        )

        show_results(cash, port_series, win_rate, buy, sell, hold, total_trades)
        metrics(port_series)
