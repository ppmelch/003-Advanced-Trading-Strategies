from libraries import *
from indicators import Indicators
from functions import Params_Indicators
from normalization import normalize_all_indicators, normalize_price


def clean_data(activo: str, intervalo: str = "15y") -> pd.DataFrame:
    """
    Download and sanitize historical OHLCV data from Yahoo Finance.

    The function fetches daily data, flattens multi-index columns if present,
    ensures the presence of a valid 'Close' column (falling back to 'Adj Close'
    or the mean of ['Open', 'High', 'Low'] when needed), keeps standard columns,
    coerces them to numeric, and drops rows with missing OHLC values.

    Parameters
    ----------
    activo : str
        Ticker symbol (e.g., "AAPL").
    intervalo : str, default="15y"
        Lookback in the form ``<int><unit>`` where unit ∈ {d, w, m, y}.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    n, u = re.match(r"(\d+)([dwmy])", intervalo.lower()).groups()
    delta = {"d": "days", "w": "weeks", "m": "months", "y": "years"}[u]
    start = dt.date.today() - relativedelta(**{delta: int(n)})

    data = yf.download(
        activo, start=start, end=dt.date.today() + dt.timedelta(1),
        interval="1d", progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    if "Close" not in data.columns or data["Close"].isna().all():
        if "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]
        else:
            data["Close"] = data[["Open", "High", "Low"]].mean(axis=1)

    if pd.isna(data.loc[data.index[-1], "Close"]):
        if "Adj Close" in data.columns and not pd.isna(data.loc[data.index[-1], "Adj Close"]):
            data.loc[data.index[-1],
                     "Close"] = data.loc[data.index[-1], "Adj Close"]
        else:
            data.loc[data.index[-1], "Close"] = data.loc[data.index[-1],
                                                         ["Open", "High", "Low"]].mean()

    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    return data


def dataset_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered dataset into train, test, and validation partitions.

    The split uses fixed proportions: 60% train, 20% test, 20% validation.

    Parameters
    ----------
    data : pd.DataFrame
        Full, time-ordered dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, test, validation) partitions, preserving order.
    """
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    validation_data = data[train_size + test_size:]

    return train_data, test_data, validation_data


def all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append a comprehensive set of technical indicators.

    Momentum: RSI(7/10/14/20), Awesome Oscillator, Williams %R, ROC, Stochastic (%K/%D)
    Volatility: ATR, Bollinger Bands, Donchian Channel, Keltner Channel
    Volume: OBV, CMF, ADI, VPT

    Parameters
    ----------
    data : pd.DataFrame
        Input OHLCV DataFrame containing at least 'Close', 'High', 'Low'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added indicator columns and NaN rows removed.
    """
    data = data.copy()
    indicators = Indicators(Params_Indicators())

    data = indicators.momentum.rsi(data)
    data = indicators.momentum.aos(data)
    data = indicators.momentum.willr(data)
    data = indicators.momentum.roc(data)
    data = indicators.momentum.stco(data)

    data = indicators.volatility.atr(data)
    data = indicators.volatility.bb(data)
    data = indicators.volatility.dchanel(data)
    data = indicators.volatility.kc(data)

    data = indicators.volume.obv(data)
    data = indicators.volume.cmf(data)
    data = indicators.volume.Acc(data)
    data = indicators.volume.vpt(data)

    data.dropna(inplace=True)
    return data


def get_signals(data: pd.DataFrame, horizon: int = 10, alpha: float = 0.01) -> pd.DataFrame:
    """
    Generate ternary trading signals based on future returns.

    Signals are derived from the forward return over a fixed horizon:
    -  1 (long)  if future_return > alpha
    -  0 (hold)  otherwise
    - -1 (short) if future_return < -alpha

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with a 'Close' column.
    horizon : int, default=10
        Look-ahead window (number of periods) to compute future price.
    alpha : float, default=0.01
        Return threshold used to assign long/short signals.

    Returns
    -------
    pd.DataFrame
        Copy with 'future_price', 'future_return', and 'signal' columns.
        The last `horizon` rows are removed to avoid look-ahead bias.
    """
    data = data.copy()
    data["future_price"] = data["Close"].shift(-horizon)
    data["future_return"] = (data["future_price"] -
                             data["Close"]) / data["Close"]
    data["signal"] = np.select(
        [data["future_return"] > alpha, data["future_return"] < -alpha],
        [1, -1], default=0
    )
    return data.dropna().iloc[:-horizon]


def all_normalization_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize indicator columns using consistent, scale-aware transformations.

    Momentum
        RSI, Stochastic → scaled to [0, 1]; AO and ROC → standardized (z-score).
        Williams %R     → shifted to [0, 1].
    Volatility
        ATR and bands (Bollinger, Donchian, Keltner) scaled relative to 'Close'.
        Relative band position features are added when applicable.
    Volume
        OBV, ADI, VPT standardized (z-score); CMF scaled from [-1, 1] to [0, 1].

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with previously computed technical indicators.

    Returns
    -------
    pd.DataFrame
        Copy with normalized indicator columns.
    """
    data = data.copy()
    normalizer = normalize_all_indicators()

    data = normalizer.Momentum().rsi(data)
    data = normalizer.Momentum().aos(data)
    data = normalizer.Momentum().willr(data)
    data = normalizer.Momentum().roc(data)
    data = normalizer.Momentum().stco(data)

    data = normalizer.Volatility().atr(data)
    data = normalizer.Volatility().bb(data)
    data = normalizer.Volatility().dchanel(data)
    data = normalizer.Volatility().kc(data)

    data = normalizer.Volume().obv(data)
    data = normalizer.Volume().cmf(data)
    data = normalizer.Volume().Acc(data)
    data = normalizer.Volume().vpt(data)

    return data


def all_normalization_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a z-scored version of 'Close' as 'Close_Z'.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Copy with a new 'Close_Z' column (original 'Close' is preserved).
    """
    data = data.copy()
    data = normalize_price.Price().close(data)
    return data


def target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split features and target from a labeled dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Labeled DataFrame containing a 'signal' column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X: Feature matrix (all columns except 'signal').
        y: Target vector (the 'signal' column).
    """
    data = data.copy()
    X = data.drop(columns=['signal'])
    y = data['signal']
    return X, y


def process_dataset(data: pd.DataFrame, alpha: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for a single split.

    Steps
    -----
    1) Compute indicators
    2) Create forward-looking trading signals
    3) Normalize indicators
    4) Add z-scored price

    Parameters
    ----------
    data : pd.DataFrame
        Raw OHLCV DataFrame.
    alpha : float, default=0.01
        Signal threshold used by `get_signals`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (data_with_signals, normalized_indicators, price_with_close_z)
    """
    data = all_indicators(data)
    data = get_signals(data, alpha=alpha)
    data_ind = all_normalization_indicators(data)
    data_price = all_normalization_price(data)
    return data, data_ind, data_price