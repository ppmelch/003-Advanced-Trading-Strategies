from libraries import *

class Metrics:
    def __init__(self, data: pd.Series):
        """
        Initializes the Metrics class with historical data.
        """
        self.data = data
        self.returns = data.pct_change(fill_method=None).dropna(
        ) if not data.empty else pd.Series(dtype=float)

    @property
    def sharpe(self) -> float:
        """
        Calculates the annualized Sharpe ratio.
        Sharpe = (annualized mean return) / (annualized standard deviation)
        Returns 0 if no data or if standard deviation is zero.
        """
        if self.returns.empty:
            return 0.0
        mean_ret = self.returns.mean()  # Average return per period
        std_ret = self.returns.std()    # Standard deviation per period
        # Annualized mean (assuming hourly data)
        annual_mean = mean_ret * (365 * 24)
        # Annualized standard deviation
        annual_std = std_ret * np.sqrt(365 * 24)
        return annual_mean / annual_std if annual_std > 0 else 0.0

    @property
    def sortino(self) -> float:
        """
        Calculates the annualized Sortino ratio.
        Sortino = (annualized mean return) / (annualized downside deviation)
        Only considers negative return volatility.
        """
        if self.returns.empty:
            return 0.0
        mean_ret = self.returns.mean()                     # Average return
        # Std deviation of negative returns
        downside_std = np.minimum(self.returns, 0).std()
        annual_mean = mean_ret * (365 * 24)               # Annualized mean
        annual_downside_std = downside_std * np.sqrt(365 * 24)
        return annual_mean / annual_downside_std if annual_downside_std > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """
        Calculates the maximum drawdown from peak to trough.
        Returns a positive value representing the largest drop.
        """
        if self.data.empty:
            return 0.0
        # Cumulative maximum up to each point
        rolling_max = self.data.cummax()
        drawdowns = (self.data - rolling_max) / \
            rolling_max   # Relative drop from peak
        return abs(drawdowns.min()) if not drawdowns.empty else 0.0

    @property
    def calmar(self) -> float:
        """
        Calculates the Calmar ratio: annualized return / maximum drawdown.
        Measures portfolio efficiency relative to risk.
        """
        if self.returns.empty:
            return 0.0
        annual_mean = self.returns.mean() * (365 * 24)   # Annualized return
        max_dd = self.max_drawdown                        # Maximum drawdown
        return annual_mean / max_dd if max_dd > 0 else 0.0

    @staticmethod
    def win_rate(closed_positions) -> float:
        """
        Calculates the proportion of winning trades.
        closed_positions: list of closed positions with a 'profit' attribute.
        """
        if not closed_positions:
            return 0.0
        n_wins = sum(
            1 for pos in closed_positions if pos.profit is not None and pos.profit > 0)
        # Percentage of profitable trades
        return n_wins / len(closed_positions)

def metrics(port_value: pd.Series) -> pd.DataFrame:
    """
    Calcula y devuelve un DataFrame con varias métricas financieras clave
    basadas en una serie temporal de valores de portafolio.
    Args:
        port_value (pd.Series): Serie temporal que contiene los valores del portafolio a lo largo del tiempo.
    Returns:
        pd.DataFrame: DataFrame que contiene las métricas calculadas.
    """
    metrics_df = pd.DataFrame({
        'Sharpe Ratio': [Metrics.sharpe(port_value)],
        'Sortino Ratio': [Metrics.sortino(port_value)],
        'Maximum Drawdown': [Metrics.max_drawdown(port_value)],
        'Calmar Ratio': [Metrics.calmar(port_value)],
        'Win Rate': [Metrics.win_rate(port_value)],
    }, index = ["Metrics"])
    return metrics_df