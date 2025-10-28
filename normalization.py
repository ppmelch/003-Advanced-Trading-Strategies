from libraries import * 

class normalize_all_indicators:
    class Momentum:
        def rsi(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            for col in ['RSI_7', 'RSI_10', 'RSI_14', 'RSI_20']:
                if col in data.columns:
                    data[col] = data[col] / 100 
            return data

        def aos(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            for col in ['Awesome_Oscillator']:
                if col in data.columns:
                    data[col] = (data[col] - data[col].mean()) / data[col].std()  
            return data

        def willr(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            for col in ['Williams_%R']:
                if col in data.columns:
                    data[col] = (data[col] + 100) / 100  
            return data

        def roc(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            for col in ['ROC']:
                if col in data.columns:
                    data[col] = (data[col] - data[col].mean()) / data[col].std()  
            return data

        def stco(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            for col in ['Stoch_K', 'Stoch_D']:
                if col in data.columns:
                    data[col] = data[col] / 100 
            return data

    class Volatility:
        def atr(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            if 'ATR' in data.columns and 'Close' in data.columns:
                data['ATR'] = data['ATR'] / data['Close']
            return data

        def bb(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            cols = ['Bollinger_Hband', 'Bollinger_Lband', 'Bollinger_Mavg']
            for col in cols:
                if col in data.columns and 'Close' in data.columns:
                    data[col] = data[col] / data['Close']
            if all(c in data.columns for c in ['Close', 'Bollinger_Lband', 'Bollinger_Hband']):
                data['Bollinger_Position'] = (
                    data['Close'] - data['Bollinger_Lband']
                ) / (data['Bollinger_Hband'] - data['Bollinger_Lband'])
            return data

        def dchanel(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            for col in ['Donchian_Hband', 'Donchian_Lband', 'Donchian_Mband']:
                if col in data.columns and 'Close' in data.columns:
                    data[col] = data[col] / data['Close']
            if all(c in data.columns for c in ['Close', 'Donchian_Lband', 'Donchian_Hband']):
                data['Donchian_Position'] = (
                    data['Close'] - data['Donchian_Lband']
                ) / (data['Donchian_Hband'] - data['Donchian_Lband'])
            return data

        def kc(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            for col in ['Keltner_Hband', 'Keltner_Lband']:
                if col in data.columns and 'Close' in data.columns:
                    data[col] = data[col] / data['Close']
            if all(c in data.columns for c in ['Close', 'Keltner_Lband', 'Keltner_Hband']):
                data['Keltner_Position'] = (
                    data['Close'] - data['Keltner_Lband']
                ) / (data['Keltner_Hband'] - data['Keltner_Lband'])
            return data

    class Volume:
        def obv(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            if 'OBV' in data.columns:
                data['OBV'] = (data['OBV'] - data['OBV'].mean()) / data['OBV'].std()
            return data

        def cmf(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            if 'CMF' in data.columns:
                data['CMF'] = (data['CMF'] + 1) / 2  
            return data

        def Acc(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            if 'ADI' in data.columns:
                data['ADI'] = (data['ADI'] - data['ADI'].mean()) / data['ADI'].std()
            return data

        def vpt(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            if 'VPT' in data.columns:
                data['VPT'] = (data['VPT'] - data['VPT'].mean()) / data['VPT'].std()
            return data
    
    class Price:
        def normaliza_all_indicators_close(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            if 'Close' in data.columns:
                data['Close'] =  normalize_all_indicators._zscore(data['Close'])
            return data
        
class normalize_price:
    class Price:
        @staticmethod
        def _zscore(series: pd.Series) -> pd.Series:
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0 or not pd.notna(std):
                std = 1
            return (series - mean) / std

        def close(self, data: pd.DataFrame) -> pd.DataFrame:
            data = data.copy()
            if 'Close' in data.columns:
                data['Close_Z'] = self._zscore(data['Close'])
            return data