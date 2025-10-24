from libraries import *
from functions import Params_Indicators


class Indicators:
    def __init__(self, params: Params_Indicators):
        self.params = params
        self.momentum = self.Momentum(params)
        self.volatility = self.Volatility(params)
        self.volume = self.Volume(params)

    class Momentum:
        def __init__(self, params: Params_Indicators):
            self.params = params

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

        def aos(self, data: pd.DataFrame) -> pd.DataFrame:
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
            data = data.copy()
            wr = WilliamsRIndicator(
                high=data['High'], low=data['Low'], close=data['Close'],
                lbp=self.params.williams_r_lbp
            )
            data['Williams_%R'] = wr.williams_r()
            return data

        def roc(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            roc = ROCIndicator(
                close=data['Close'], window=self.params.roc_window, fillna=False)
            data['ROC'] = roc.roc()
            return data

        def stco(self, data: pd.DataFrame) -> pd.DataFrame:
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
        def __init__(self, params: Params_Indicators):
            self.params = params

        def atr(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            atr = AverageTrueRange(
                high=data['High'], low=data['Low'], close=data['Close'],
                window=self.params.atr_window, fillna=False
            )
            data['ATR'] = atr.average_true_range()
            return data

        def bb(self, data: pd.DataFrame) -> pd.DataFrame:
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
        def __init__(self, params: Params_Indicators):
            self.params = params

        def obv(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            obv = OnBalanceVolumeIndicator(
                close=data['Close'], volume=data['Volume'],
                fillna=False
            )
            data['OBV'] = obv.on_balance_volume()
            return data

        def cmf(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            cmf = ChaikinMoneyFlowIndicator(
                high=data['High'], low=data['Low'], close=data['Close'],
                volume=data['Volume'], window=self.params.cmf_window,
                fillna=False
            )
            data['CMF'] = cmf.chaikin_money_flow()
            return data

        def Acc(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            adi = AccDistIndexIndicator(
                high=data['High'], low=data['Low'],
                close=data['Close'], volume=data['Volume'],
                fillna=False
            )
            data['ADI'] = adi.acc_dist_index()
            return data

        def vpt(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            vpt = VolumePriceTrendIndicator(
                close=data['Close'], volume=data['Volume'],
                fillna=False
            )
            data['VPT'] = vpt.volume_price_trend()
            return data

