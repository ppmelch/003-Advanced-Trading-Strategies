from libraries import *
from indicators import Indicators
from functions import Params_Indicators


def clean_data(activo: str, intervalo: str = "15y") -> pd.DataFrame:
    """
    Downloads and cleans stock data for a given asset and interval.
    Parameters:
    activo : str
        Stock ticker symbol.
    intervalo : str
        Time interval for data retrieval (e.g., '15y' for 15 years).
    Returns:
    pd.DataFrame
        Cleaned DataFrame with stock data.
    """
    n, u = re.match(r"(\d+)([dwmy])", intervalo.lower()).groups()
    delta = {"d": "days", "w": "weeks", "m": "months", "y": "years"}[u]
    start = dt.date.today() - relativedelta(**{delta: int(n)})
    data = yf.download(activo, start=start, end=dt.date.today(
    )+dt.timedelta(1), interval="1d", progress=False).reset_index()
    data.rename(columns={"Datetime": "Date"}, inplace=True)
    return data[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()


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
    n = len(data)
    train_size = int(n * 0.6)
    test_size = train_size + int(n * 0.2)

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
