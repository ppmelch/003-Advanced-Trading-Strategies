from libraries import *
from functions import Params_Indicators


class Indicators:
    def __init__(self, params: Params_Indicators):
        self.params = params
        self.momentum = self.Momentum(self)
        self.volatility = self.Volatility(self)
        self.volume = self.Volume(self)

    # 8 Indicadores de Momentum 
    class Momentum:
        def __init__(self, data: pd.DataFrame, params: Params_Indicators):
            self.data = data
            self.params = params

        def rsi(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()

        def stochastic_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
        def macd(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
        def cci(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
        def williams_r(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
        def roc(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
        def awesome_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
        def cmi(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
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


