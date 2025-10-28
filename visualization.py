from functions import Config
from libraries import *


def plot_test_validation(port_value_test: pd.Series, port_value_val: pd.Series) -> None:
    """
    Plot the test and validation portfolio curves continuously.

    The test portfolio is rescaled to start at the initial capital (1,000,000),
    and the validation portfolio begins at the final value of the test portfolio.

    Parameters
    ----------
    port_value_test : pd.Series
        Time series of portfolio values from the test phase.
    port_value_val : pd.Series
        Time series of portfolio values from the validation phase.

    Returns
    -------
    None
        Displays the combined test and validation performance plot.
    """
    plt.figure(figsize=(12, 6))

    test_scaled = port_value_test / \
        port_value_test.iloc[0] * Config.initial_capital
    val_scaled = port_value_val / port_value_val.iloc[0] * test_scaled.iloc[-1]

    x_test = range(len(test_scaled))
    x_val = range(len(test_scaled), len(test_scaled) + len(val_scaled))

    plt.plot(x_test, test_scaled, label="Test", color="royalblue", lw=2)
    plt.plot(x_val, val_scaled, label="Validation", color="orange", lw=2)

    plt.title("Test + Validation Portfolio", fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_single_split(series: pd.Series, title: str) -> None:
    """
    Plot a single portfolio time series.

    Parameters
    ----------
    series : pd.Series
        Portfolio value series to plot.
    title : str
        Plot title.

    Returns
    -------
    None
        Displays the portfolio value plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series.values, marker="o", markersize=2,
             linewidth=1.5, label=title)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_test_and_validation(test_series: pd.Series, val_series: pd.Series, title: str) -> None:
    """
    Plot test and validation portfolio curves on the same axis.

    The validation curve starts immediately after the test curve, 
    creating a continuous time progression between both.

    Parameters
    ----------
    test_series : pd.Series
        Portfolio values from the test phase.
    val_series : pd.Series
        Portfolio values from the validation phase.
    title : str
        Plot title.

    Returns
    -------
    None
        Displays the combined test and validation plot.
    """
    plt.figure(figsize=(12, 6))
    x_test = range(len(test_series))
    x_val = range(len(test_series), len(test_series) + len(val_series))

    plt.plot(x_test, test_series.values, label="Test",
             linewidth=1.8, marker="o", markersize=2)
    plt.plot(x_val, val_series.values, label="Validation",
             linewidth=1.8, marker="o", markersize=2)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()