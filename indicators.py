from libraries import *
from functions import Params_Indicators


@dataclass
class Indicadores:
    params: Params_Indicators

    # --- Momentum Indicators ---
    def rsi(self, data: pd.DataFrame) -> pd.Series:
        return ta.momentum.RSIIndicator(
            close=data['Close'], window=self.params.rsi_window
        ).rsi()

    def stochastic_oscillator(self, data: pd.DataFrame) -> pd.Series:
        return ta.momentum.StochasticOscillator(
            high=data['High'], low=data['Low'], close=data['Close'],
            window=self.params.stoch_osc_window,
            smooth_window=self.params.stoch_osc_smooth
        ).stoch()

    def macd(self, data: pd.DataFrame) -> pd.Series:
        return ta.trend.MACD(
            close=data['Close'],
            window_fast=self.params.macd_fast,
            window_slow=self.params.macd_slow,
            window_sign=self.params.macd_signal
        ).macd()

    def cci(self, data: pd.DataFrame, period: str = "cci_window") -> pd.Series:
        return ta.trend.CCIIndicator(
            high=data['High'], low=data['Low'], close=data['Close'],
            window=getattr(self.params, period)
        ).cci()

    def williams_r(self, data: pd.DataFrame) -> pd.Series:
        return ta.momentum.WilliamsRIndicator(
            high=data['High'], low=data['Low'], close=data['Close'],
            lbp=self.params.williams_r_lbp
        ).williams_r()

    def roc(self, data: pd.DataFrame) -> pd.Series:
        return ta.momentum.ROCIndicator(
            close=data['Close'], window=self.params.roc_window
        ).roc()

    def awesome_oscillator(self, data: pd.DataFrame) -> pd.Series:
        return ta.momentum.AwesomeOscillator(
            high=data['High'], low=data['Low'],
            window1=self.params.awesome_window1,
            window2=self.params.awesome_window2
        ).awesome_oscillator()

    def cmi(self, data: pd.DataFrame) -> pd.Series:
        return ta.momentum.ClassicMomentumIndicator(
            close=data['Close'], window=self.params.cmi_window
        ).cmi()

    # --- Volatility Indicators ---
    def atr(self, data: pd.DataFrame) -> pd.Series:
        return ta.volatility.AverageTrueRange(
            high=data['High'], low=data['Low'], close=data['Close'],
            window=self.params.atr_window
        ).average_true_range()

    def bollinger_bands(self, data: pd.DataFrame, period: str = "bollinger_window") -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(
            close=data['Close'], window=getattr(self.params, period), window_dev=self.params.bollinger_dev
        )
        return pd.DataFrame({
            'bollinger_upper': bb.bollinger_hband(),
            'bollinger_lower': bb.bollinger_lband()
        })

    def donchian_channels(self, data: pd.DataFrame, period: str = "donchian_window") -> pd.DataFrame:
        dc = ta.volatility.DonchianChannel(
            high=data['High'], low=data['Low'], window=getattr(
                self.params, period)
        )
        return pd.DataFrame({
            'donchian_upper': dc.donchian_channel_hband(),
            'donchian_lower': dc.donchian_channel_lband()
        })

    def keltner_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        kc = ta.volatility.KeltnerChannel(
            high=data['High'], low=data['Low'], close=data['Close'],
            window=self.params.keltner_window, window_atr=self.params.keltner_atr
        )
        return pd.DataFrame({
            'keltner_upper': kc.keltner_channel_hband(),
            'keltner_lower': kc.keltner_channel_lband()
        })

    def chaikin_volatility(self, data: pd.DataFrame) -> pd.Series:
        return ta.volatility.ChaikinVolatility(
            high=data['High'], low=data['Low'], window=self.params.chaikin_vol_window
        ).chaikin_volatility()

    def ulcer_index(self, data: pd.DataFrame) -> pd.Series:
        return ta.volatility.UlcerIndex(
            close=data['Close'], window=self.params.ulcer_index_window
        ).ulcer_index()

    # --- Volume Indicators ---
    def obv(self, data: pd.DataFrame) -> pd.Series:
        return ta.volume.OnBalanceVolumeIndicator(
            close=data['Close'], volume=data['Volume']
        ).on_balance_volume()

    def cmf(self, data: pd.DataFrame) -> pd.Series:
        return ta.volume.ChaikinMoneyFlowIndicator(
            high=data['High'], low=data['Low'], close=data['Close'],
            volume=data['Volume'], window=self.params.cmf_window
        ).chaikin_money_flow()

    def vpt(self, data: pd.DataFrame) -> pd.Series:
        return ta.volume.VolumePriceTrendIndicator(
            close=data['Close'], volume=data['Volume']
        ).volume_price_trend()

    def mfi(self, data: pd.DataFrame) -> pd.Series:
        return ta.volume.MFIIndicator(
            high=data['High'], low=data['Low'], close=data['Close'],
            volume=data['Volume'], window=self.params.money_flow_index_window
        ).money_flow_index()

    def force_index(self, data: pd.DataFrame) -> pd.Series:
        return ta.volume.ForceIndexIndicator(
            close=data['Close'], volume=data['Volume'], window=self.params.force_index_window
        ).force_index()