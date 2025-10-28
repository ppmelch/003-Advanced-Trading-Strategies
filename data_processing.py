from libraries import *
from indicators import Indicators
from functions import Params_Indicators
from dateutil.relativedelta import relativedelta

from normalization import normalize_all_indicators, normalize_price

def clean_data(activo: str, intervalo: str = "15y") -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance.
    If the most recent candle lacks a 'Close' (e.g., ongoing trading day),
    it uses the last available price (from 'Adj Close' or 'Open'/'High'/'Low').
    """
    n, u = re.match(r"(\d+)([dwmy])", intervalo.lower()).groups()
    delta = {"d": "days", "w": "weeks", "m": "months", "y": "years"}[u]
    start = dt.date.today() - relativedelta(**{delta: int(n)})

    data = yf.download(
        activo, start=start, end=dt.date.today() + dt.timedelta(1),
        interval="1d", progress=False
    )

    # --- Flatten multilevel columns if needed ---
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # --- If Close column has missing or 0 rows, fill with last valid price ---
    if "Close" not in data.columns or data["Close"].isna().all():
        if "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]
        else:
            # fallback: average of open/high/low
            data["Close"] = data[["Open", "High", "Low"]].mean(axis=1)

    # --- Replace NaNs in last row if the market hasn’t closed ---
    if pd.isna(data.loc[data.index[-1], "Close"]):
        # Use Adj Close if available, else mean of OHLC
        if "Adj Close" in data.columns and not pd.isna(data.loc[data.index[-1], "Adj Close"]):
            data.loc[data.index[-1], "Close"] = data.loc[data.index[-1], "Adj Close"]
        else:
            data.loc[data.index[-1], "Close"] = data.loc[data.index[-1], ["Open", "High", "Low"]].mean()

    # --- Keep only standard columns ---
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # --- Ensure numeric columns and drop NaNs ---
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    return data

def dataset_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training, testing, and validation sets.

    Parameters:
    data : pd.DataFrame
        Complete dataset to split.
    train : float
        Fraction of data to use for training.
    test : float
        Fraction of data to use for testing.
    validation : float
        Fraction of data to use for validation.

    Returns:
    tuple
        train_data, test_data, validation_data
    """
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    validation_data = data[train_size + test_size:]

    return train_data, test_data, validation_data


def all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all technical indicators and adds them to the DataFrame.
    Parameters:
    data : pd.DataFrame
        Input data containing 'Close', 'High', and 'Low' columns.
    Returns: pd.DataFrame
        DataFrame with added technical indicators and NaN values removed.
    """
    data = data.copy()

    indicators = Indicators(Params_Indicators())

    # --- Momentum Indicators ---
    data = indicators.momentum.rsi(data)
    data = indicators.momentum.aos(data)
    data = indicators.momentum.willr(data)
    data = indicators.momentum.roc(data)
    data = indicators.momentum.stco(data)

    # --- Volatility Indicators ---
    data = indicators.volatility.atr(data)
    data = indicators.volatility.bb(data)
    data = indicators.volatility.dchanel(data)
    data = indicators.volatility.kc(data)

    # --- Volume Indicators ---
    data = indicators.volume.obv(data)
    data = indicators.volume.cmf(data)
    data = indicators.volume.Acc(data)
    data = indicators.volume.vpt(data)

    data.dropna(inplace=True)
    return data

def get_signals(data: pd.DataFrame, horizon: int = 5, alpha: float = 0.02) -> pd.DataFrame:
    """
    Generate trading signals based on future price movements.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain at least a 'Close' column with price data.
    horizon : int, default=5
        Number of periods ahead to calculate future returns.
    alpha : float, default=0.02
        Threshold (in decimal) for generating buy/sell signals.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'future_price': price after horizon periods
        - 'future_return': percentage change over horizon
        - 'signal': {1 = long, 0 = hold, -1 = short}
    """
    data = data.copy()

    data["future_price"] = data["Close"].shift(-horizon)
    data["future_return"] = (data["future_price"] - data["Close"]) / data["Close"]

    data["signal"] = np.where(
        data["future_return"] > alpha, 1,
        np.where(data["future_return"] < -alpha, -1, 0)
    )

    data = data.iloc[:-horizon].copy()

    return data

def all_normalization_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all technical indicators in the DataFrame.

    Parameters:
    data : pd.DataFrame
        Input data containing technical indicators.

    Returns: pd.DataFrame
        DataFrame with normalized technical indicators.
    """
    data = data.copy()

    normalizer = normalize_all_indicators()

    # --- Momentum Indicators ---
    data = normalizer.Momentum().rsi(data)
    data = normalizer.Momentum().aos(data)
    data = normalizer.Momentum().willr(data)
    data = normalizer.Momentum().roc(data)
    data = normalizer.Momentum().stco(data)

    # --- Volatility Indicators ---
    data = normalizer.Volatility().atr(data)
    data = normalizer.Volatility().bb(data)
    data = normalizer.Volatility().dchanel(data)
    data = normalizer.Volatility().kc(data)

    # --- Volume Indicators ---
    data = normalizer.Volume().obv(data)
    data = normalizer.Volume().cmf(data)
    data = normalizer.Volume().Acc(data)
    data = normalizer.Volume().vpt(data)

    return data


def all_normalization_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'Close' price in the DataFrame.

    Parameters:
    data : pd.DataFrame
        Input data containing 'Close' price.

    Returns: pd.DataFrame
        DataFrame with normalized 'Close' price.
    """
    data = data.copy()

    data = normalize_price.Price().close(data)
    return data



def target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates features and target variable from the DataFrame.

    Parameters:
    data : pd.DataFrame
        Input data containing features and 'signal' target column.

    Returns: tuple
        X : pd.DataFrame
            Features DataFrame (all columns except 'signal').
        y : pd.Series
            Target Series ('signal' column).
    """
    data = data.copy()

    X = data.drop(columns=['signal'])
    y = data['signal']

    return X, y


def process_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: tuple
        data : pd.DataFrame -> base data con señales
        data_indicators : pd.DataFrame -> indicadores normalizados (con 'signal')
        data_price : pd.DataFrame -> precio normalizado (sin leakage)
    """
    data = all_indicators(data)
    data = get_signals(data)

    data_indicators = all_normalization_indicators(data).copy()
    data_price = all_normalization_price(data).copy()

    data_indicators.dropna(inplace=True)
    data_price.dropna(inplace=True)

    return data, data_indicators, data_price

