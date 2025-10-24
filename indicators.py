from libraries import *
from functions import Params_Indicators
from ta.momentum import RSIIndicator, WilliamsRIndicator, ROCIndicator, AwesomeOscillatorIndicator, StochasticOscillator
from ta.volatility import BollingerBands, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator



class Indicators:
    def __init__(self, params: Params_Indicators):
        self.params = params
        self.momentum = self.Momentum(params)
        self.volatility = self.Volatility(self, params)
        self.volume = self.Volume(self, params)


    # 8 Indicadores de Momentum
    class Momentum:
        def __init__(self, params: Params_Indicators):
            self.params = params

        # RSI
        def rsi(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            data[f'RSI_7'] = RSIIndicator(
                data['Close'], window=self.params.rsi_7_window).rsi()
            data[f'RSI_10'] = RSIIndicator(
                data['Close'], window=self.params.rsi_10_window).rsi()
            data[f'RSI_14'] = RSIIndicator(
                data['Close'], window=self.params.rsi_14_window).rsi()
            data[f'RSI_20'] = RSIIndicator(
                data['Close'], window=self.params.rsi_20_window).rsi()
            return data

        # Awesome Oscillator
        def awesome_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            ao = AwesomeOscillatorIndicator(
                high=data['High'],
                low=data['Low'],
                window1=self.params.awe_window1,
                window2=self.params.awe_window2,
                fillna=False
            )
            data['Awesome_Oscillator'] = ao.awesome_oscillator()
            return data

        # Williams %R
        def willr(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            wr = WilliamsRIndicator(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                lbp=self.params.williams_r_lbp
            )
            data['Williams_%R'] = wr.williams_r()
            return data

        # ROC
        def roc(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            roc = ROCIndicator(
                close=data['Close'],
                window=self.params.roc_window,
                fillna=False
            )
            data['ROC'] = roc.roc()
            return data

        # Stochastic Oscillator
        def stochastic_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            stoch = StochasticOscillator(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                window=self.params.stoch_osc_window,
                smooth_window=self.params.stoch_osc_smooth,
                fillna=False
            )
            data['Stoch_K'] = stoch.stoch()
            data['Stoch_D'] = stoch.stoch_signal()
            return data

    # 8 de volatilidad

    class Volatility:
        def __init__(self, data: pd.DataFrame, params: Params_Indicators):
            self.data = data
            self.params = params

        def atr(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()

        def bollinger_bandwidth(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()

        def donchian_channel(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            return data

        def chaikin_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()

        def ulcer_index(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            return data

        def keltner_channel(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            return data

        def volatility_stop(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            return data

        def parabolic_sar(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            return data

    # 4 de volumen
    class Volume:
        def __init__(self, data: pd.DataFrame, params: Params_Indicators):
            self.data = data
            self.params = params

        def on_balance_volume(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()

        def chaikin_money_flow(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()

        def money_flow_index(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()

        def volume_price_trend(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            return data
