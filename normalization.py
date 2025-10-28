from libraries import *

class normalize_all_indicators:
    """
    Normalization routines for groups of technical indicators.

    This class groups inner classes by indicator type (Momentum, Volatility,
    Volume, Price) and exposes methods that normalize expected columns in a
    pandas DataFrame in-place (on a copy) and return the normalized DataFrame.
    """

    class Momentum:
        """
        Normalization methods for momentum-related indicators.

        Expected columns (if present) will be normalized:
        - RSI_7, RSI_10, RSI_14, RSI_20  → scaled to [0, 1] by dividing by 100.
        - Awesome_Oscillator, ROC         → standardized to zero mean and unit variance.
        - Williams_%R                     → shifted/scaled from [-100, 0] to [0, 1].
        - Stoch_K, Stoch_D                → scaled to [0, 1] by dividing by 100.
        """

        def rsi(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize Relative Strength Index columns by scaling to [0, 1].

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing RSI columns.

            Returns
            -------
            pd.DataFrame
                Copy of the input with normalized RSI columns.
            """
            data = data.copy()
            for col in ['RSI_7', 'RSI_10', 'RSI_14', 'RSI_20']:
                if col in data.columns:
                    data[col] = data[col] / 100
            return data

        def aos(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Standardize Awesome Oscillator to zero mean and unit variance.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'Awesome_Oscillator'.

            Returns
            -------
            pd.DataFrame
                Copy with standardized 'Awesome_Oscillator'.
            """
            data = data.copy()
            for col in ['Awesome_Oscillator']:
                if col in data.columns:
                    data[col] = (data[col] - data[col].mean()) / \
                        data[col].std()
            return data

        def willr(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize Williams %R from [-100, 0] to [0, 1].

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'Williams_%R'.

            Returns
            -------
            pd.DataFrame
                Copy with normalized 'Williams_%R'.
            """
            data = data.copy()
            for col in ['Williams_%R']:
                if col in data.columns:
                    data[col] = (data[col] + 100) / 100
            return data

        def roc(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Standardize Rate of Change (ROC) to zero mean and unit variance.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'ROC'.

            Returns
            -------
            pd.DataFrame
                Copy with standardized 'ROC'.
            """
            data = data.copy()
            for col in ['ROC']:
                if col in data.columns:
                    data[col] = (data[col] - data[col].mean()) / \
                        data[col].std()
            return data

        def stco(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize stochastic oscillator lines to [0, 1].

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'Stoch_K' and/or 'Stoch_D'.

            Returns
            -------
            pd.DataFrame
                Copy with normalized stochastic columns.
            """
            data = data.copy()
            for col in ['Stoch_K', 'Stoch_D']:
                if col in data.columns:
                    data[col] = data[col] / 100
            return data

    class Volatility:
        """
        Normalization methods for volatility-related indicators.

        Expected behavior:
        - ATR is scaled by 'Close'.
        - Bollinger/Donchian/Keltner bands are scaled by 'Close'.
        - Relative position features ('*_Position') are computed in [0, 1] when
          the corresponding bands are available.
        """

        def atr(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize Average True Range (ATR) by dividing by 'Close'.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'ATR' and 'Close'.

            Returns
            -------
            pd.DataFrame
                Copy with normalized 'ATR'.
            """
            data = data.copy()
            if 'ATR' in data.columns and 'Close' in data.columns:
                data['ATR'] = data['ATR'] / data['Close']
            return data

        def bb(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize Bollinger Bands by 'Close' and compute relative position.

            If 'Close', 'Bollinger_Lband', and 'Bollinger_Hband' are present,
            adds 'Bollinger_Position' in [0, 1] as the relative price within bands.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame with Bollinger columns and 'Close'.

            Returns
            -------
            pd.DataFrame
                Copy with normalized Bollinger columns and optional position.
            """
            data = data.copy()
            cols = ['Bollinger_Hband', 'Bollinger_Lband', 'Bollinger_Mavg']
            for col in cols:
                if col in data.columns and 'Close' in data.columns:
                    data[col] = data[col] / data['Close']
            if all(c in data.columns for c in ['Close', 'Bollinger_Lband', 'Bollinger_Hband']):
                denom = (data['Bollinger_Hband'] - data['Bollinger_Lband'])
                data['Bollinger_Position'] = (
                    data['Close'] - data['Bollinger_Lband']) / denom.replace(0, np.nan)
            return data

        def dchanel(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize Donchian Channels by 'Close' and compute relative position.

            If 'Close', 'Donchian_Lband', and 'Donchian_Hband' are present,
            adds 'Donchian_Position' in [0, 1] as the relative price within bands.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame with Donchian columns and 'Close'.

            Returns
            -------
            pd.DataFrame
                Copy with normalized Donchian columns and optional position.
            """
            data = data.copy()
            for col in ['Donchian_Hband', 'Donchian_Lband', 'Donchian_Mband']:
                if col in data.columns and 'Close' in data.columns:
                    data[col] = data[col] / data['Close']
            if all(c in data.columns for c in ['Close', 'Donchian_Lband', 'Donchian_Hband']):
                denom = (data['Donchian_Hband'] - data['Donchian_Lband'])
                data['Donchian_Position'] = (
                    data['Close'] - data['Donchian_Lband']) / denom.replace(0, np.nan)
            return data

        def kc(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize Keltner Channels by 'Close' and compute relative position.

            If 'Close', 'Keltner_Lband', and 'Keltner_Hband' are present,
            adds 'Keltner_Position' in [0, 1] as the relative price within bands.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame with Keltner columns and 'Close'.

            Returns
            -------
            pd.DataFrame
                Copy with normalized Keltner columns and optional position.
            """
            data = data.copy()
            for col in ['Keltner_Hband', 'Keltner_Lband']:
                if col in data.columns and 'Close' in data.columns:
                    data[col] = data[col] / data['Close']
            if all(c in data.columns for c in ['Close', 'Keltner_Lband', 'Keltner_Hband']):
                denom = (data['Keltner_Hband'] - data['Keltner_Lband'])
                data['Keltner_Position'] = (
                    data['Close'] - data['Keltner_Lband']) / denom.replace(0, np.nan)
            return data

    class Volume:
        """
        Normalization methods for volume-related indicators.

        Expected behavior:
        - OBV, ADI, VPT are standardized to zero mean and unit variance.
        - CMF is linearly scaled to [0, 1] from its native [-1, 1].
        """

        def obv(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Standardize On-Balance Volume (OBV) to zero mean and unit variance.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'OBV'.

            Returns
            -------
            pd.DataFrame
                Copy with standardized 'OBV'.
            """
            data = data.copy()
            if 'OBV' in data.columns:
                data['OBV'] = (data['OBV'] - data['OBV'].mean()
                               ) / data['OBV'].std()
            return data

        def cmf(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Normalize Chaikin Money Flow (CMF) from [-1, 1] to [0, 1].

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'CMF'.

            Returns
            -------
            pd.DataFrame
                Copy with normalized 'CMF'.
            """
            data = data.copy()
            if 'CMF' in data.columns:
                data['CMF'] = (data['CMF'] + 1) / 2
            return data

        def Acc(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Standardize Accumulation/Distribution Index (ADI).

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'ADI'.

            Returns
            -------
            pd.DataFrame
                Copy with standardized 'ADI'.
            """
            data = data.copy()
            if 'ADI' in data.columns:
                data['ADI'] = (data['ADI'] - data['ADI'].mean()
                               ) / data['ADI'].std()
            return data

        def vpt(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Standardize Volume Price Trend (VPT).

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'VPT'.

            Returns
            -------
            pd.DataFrame
                Copy with standardized 'VPT'.
            """
            data = data.copy()
            if 'VPT' in data.columns:
                data['VPT'] = (data['VPT'] - data['VPT'].mean()
                               ) / data['VPT'].std()
            return data

    class Price:
        """
        Normalization methods for price-related features.
        """

        def normaliza_all_indicators_close(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Z-score normalize the 'Close' column in-place.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'Close'.

            Returns
            -------
            pd.DataFrame
                Copy with z-scored 'Close'.
            """
            data = data.copy()
            if 'Close' in data.columns:
                data['Close'] = normalize_price.Price._zscore(data['Close'])
            return data


class normalize_price:
    """
    Normalization utilities focused on raw price series.
    """

    class Price:
        """
        Price-specific normalization helpers.
        """

        @staticmethod
        def _zscore(series: pd.Series) -> pd.Series:
            """
            Compute z-score with population standard deviation (ddof=0).

            Parameters
            ----------
            series : pd.Series
                Input series to standardize.

            Returns
            -------
            pd.Series
                Z-scored series. If the standard deviation is zero or NaN,
                a value of 1 is used to avoid division by zero.
            """
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0 or not pd.notna(std):
                std = 1
            return (series - mean) / std

        def close(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Add a z-scored version of 'Close' as 'Close_Z'.

            Parameters
            ----------
            data : pd.DataFrame
                Input DataFrame containing 'Close'.

            Returns
            -------
            pd.DataFrame
                Copy with a new column 'Close_Z'.
            """
            data = data.copy()
            if 'Close' in data.columns:
                data['Close_Z'] = self._zscore(data['Close'])
            return data
