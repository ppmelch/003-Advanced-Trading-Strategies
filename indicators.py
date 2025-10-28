from libraries import *
from functions import Params_Indicators


class Indicators:
    """
    Compute technical indicators for momentum, volatility, and volume analysis.

    This class groups several types of technical indicators commonly used in
    financial time series analysis. It uses parameter values from a
    `Params_Indicators` dataclass to control window sizes and other settings.

    Attributes
    ----------
    params : Params_Indicators
        Configuration parameters for all indicator calculations.
    momentum : Indicators.Momentum
        Subclass handling momentum-related indicators.
    volatility : Indicators.Volatility
        Subclass handling volatility-related indicators.
    volume : Indicators.Volume
        Subclass handling volume-related indicators.
    """

    def __init__(self, params: Params_Indicators):
        self.params = params
        self.momentum = self.Momentum(params)
        self.volatility = self.Volatility(params)
        self.volume = self.Volume(params)

    class Momentum:
        """
        Compute momentum-based technical indicators such as RSI, ROC, and Stochastic Oscillator.
        """

        def __init__(self, params: Params_Indicators):
            self.params = params

        def rsi(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute multiple RSI (Relative Strength Index) values.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'Close' price.

            Returns
            -------
            pd.DataFrame
                Copy of input DataFrame with added RSI columns.
            """
            data = data.copy()
            data['RSI_7'] = RSIIndicator(
                data['Close'], window=self.params.rsi_7_window).rsi()
            data['RSI_10'] = RSIIndicator(
                data['Close'], window=self.params.rsi_10_window).rsi()
            data['RSI_14'] = RSIIndicator(
                data['Close'], window=self.params.rsi_14_window).rsi()
            data['RSI_20'] = RSIIndicator(
                data['Close'], window=self.params.rsi_20_window).rsi()
            return data

        def aos(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute the Awesome Oscillator (AO).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High' and 'Low' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'Awesome_Oscillator' column added.
            """
            data = data.copy()
            ao = AwesomeOscillatorIndicator(
                high=data['High'], low=data['Low'],
                window1=self.params.awe_window1,
                window2=self.params.awe_window2,
                fillna=False
            )
            data['Awesome_Oscillator'] = ao.awesome_oscillator()
            return data

        def willr(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute the Williams %R indicator.

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High', 'Low', and 'Close' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'Williams_%R' column added.
            """
            data = data.copy()
            wr = WilliamsRIndicator(
                high=data['High'], low=data['Low'], close=data['Close'],
                lbp=self.params.williams_r_lbp
            )
            data['Williams_%R'] = wr.williams_r()
            return data

        def roc(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute the Rate of Change (ROC) indicator.

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'Close' column.

            Returns
            -------
            pd.DataFrame
                Copy with 'ROC' column added.
            """
            data = data.copy()
            roc = ROCIndicator(
                close=data['Close'], window=self.params.roc_window, fillna=False)
            data['ROC'] = roc.roc()
            return data

        def stco(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute the Stochastic Oscillator (%K and %D).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High', 'Low', and 'Close' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'Stoch_K' and 'Stoch_D' columns added.
            """
            data = data.copy()
            stoch = StochasticOscillator(
                high=data['High'], low=data['Low'], close=data['Close'],
                window=self.params.stoch_osc_window,
                smooth_window=self.params.stoch_osc_smooth,
                fillna=False
            )
            data['Stoch_K'] = stoch.stoch()
            data['Stoch_D'] = stoch.stoch_signal()
            return data

    class Volatility:
        """
        Compute volatility-based indicators such as ATR, Bollinger Bands, 
        Donchian Channels, and Keltner Channels.
        """

        def __init__(self, params: Params_Indicators):
            self.params = params

        def atr(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute the Average True Range (ATR).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High', 'Low', and 'Close' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'ATR' column added.
            """
            data = data.copy()
            atr = AverageTrueRange(
                high=data['High'], low=data['Low'], close=data['Close'],
                window=self.params.atr_window, fillna=False
            )
            data['ATR'] = atr.average_true_range()
            return data

        def bb(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute Bollinger Bands (Middle, High, and Low).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'Close' column.

            Returns
            -------
            pd.DataFrame
                Copy with Bollinger Bands columns added.
            """
            data = data.copy()
            bb = BollingerBands(
                close=data['Close'],
                window=self.params.bollinger_window,
                window_dev=self.params.bollinger_dev,
                fillna=False
            )
            data['Bollinger_Mavg'] = bb.bollinger_mavg()
            data['Bollinger_Hband'] = bb.bollinger_hband()
            data['Bollinger_Lband'] = bb.bollinger_lband()
            return data

        def dchanel(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute Donchian Channel (High, Low, and Middle bands).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High', 'Low', and 'Close' columns.

            Returns
            -------
            pd.DataFrame
                Copy with Donchian Channel columns added.
            """
            data = data.copy()
            dc = DonchianChannel(
                high=data['High'], low=data['Low'], close=data['Close'],
                window=self.params.donchian_window, fillna=False
            )
            data['Donchian_Hband'] = dc.donchian_channel_hband()
            data['Donchian_Lband'] = dc.donchian_channel_lband()
            data['Donchian_Mband'] = dc.donchian_channel_mband()
            return data

        def kc(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute Keltner Channel (High and Low bands).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High', 'Low', and 'Close' columns.

            Returns
            -------
            pd.DataFrame
                Copy with Keltner Channel columns added.
            """
            data = data.copy()
            kc = KeltnerChannel(
                high=data['High'], low=data['Low'], close=data['Close'],
                window=self.params.keltner_window,
                window_atr=self.params.keltner_atr,
                fillna=False
            )
            data['Keltner_Hband'] = kc.keltner_channel_hband()
            data['Keltner_Lband'] = kc.keltner_channel_lband()
            return data

    class Volume:
        """
        Compute volume-based technical indicators such as OBV, CMF, ADI, and VPT.
        """

        def __init__(self, params: Params_Indicators):
            self.params = params

        def obv(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute On-Balance Volume (OBV).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'Close' and 'Volume' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'OBV' column added.
            """
            data = data.copy()
            obv = OnBalanceVolumeIndicator(
                close=data['Close'], volume=data['Volume'], fillna=False
            )
            data['OBV'] = obv.on_balance_volume()
            return data

        def cmf(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute Chaikin Money Flow (CMF).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High', 'Low', 'Close', and 'Volume' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'CMF' column added.
            """
            data = data.copy()
            cmf = ChaikinMoneyFlowIndicator(
                high=data['High'], low=data['Low'], close=data['Close'],
                volume=data['Volume'], window=self.params.cmf_window, fillna=False
            )
            data['CMF'] = cmf.chaikin_money_flow()
            return data

        def Acc(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute Accumulation/Distribution Index (ADI).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'High', 'Low', 'Close', and 'Volume' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'ADI' column added.
            """
            data = data.copy()
            adi = AccDistIndexIndicator(
                high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'],
                fillna=False
            )
            data['ADI'] = adi.acc_dist_index()
            return data

        def vpt(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Compute Volume Price Trend (VPT).

            Parameters
            ----------
            data : pd.DataFrame
                DataFrame with 'Close' and 'Volume' columns.

            Returns
            -------
            pd.DataFrame
                Copy with 'VPT' column added.
            """
            data = data.copy()
            vpt = VolumePriceTrendIndicator(
                close=data['Close'], volume=data['Volume'], fillna=False
            )
            data['VPT'] = vpt.volume_price_trend()
            return data
