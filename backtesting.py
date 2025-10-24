from libraries import *
from metrics import Metrics
from indicators import Indicators
from functions import Position, Config , Params_Indicators, get_portfolio_value


#ARREGLAR ESTO PARA QUE USE LOS INDICADORES TECNICOS

def backtest(data: pd.DataFrame, params: Params_Indicators, initial_cash: float = None) -> tuple[list, dict, float]:
    """
    Backtest usando señales derivadas de indicadores técnicos (RSI, Momentum, Volatility)
    con filtro de volatilidad y gestión de posiciones (long/short) incluyendo Borrow Rate.
    """

    # --- 1. Preparar DataFrame de indicadores ---
    indicators = Indicators(params=params)
    data_ind = indicators.get_data(data)
    data_ind = data_ind.reset_index(drop=True)

    # --- 2. Señales ---
    # RSI
    buy_rsi = data_ind['RSI'] < 30
    sell_rsi = data_ind['RSI'] > 70

    # Momentum (ejemplo usando ROC y CMI)
    buy_momentum = (data_ind['ROC'] > 0) & (data_ind['CMI'] > 0)
    sell_momentum = (data_ind['ROC'] < 0) & (data_ind['CMI'] < 0)

    # Volatility filter (Bollinger Bands)
    buy_volatility = data_ind['Close'] < data_ind['bollinger_lower']
    sell_volatility = data_ind['Close'] > data_ind['bollinger_upper']

    # --- Volatility cuantil como filtro de mercado estable ---
    vol_window = 20
    vol_quantile = 0.5
    vol = data_ind['Close'].rolling(vol_window).std()
    low_vol = vol < vol.quantile(vol_quantile)

    # --- Combinar señales ---
    historic = data_ind.copy()
    historic['buy_signal'] = ((buy_rsi + 2 * buy_momentum) >= 2) & low_vol
    historic['sell_signal'] = ((sell_rsi + 2 * sell_momentum) >= 2) & low_vol
    historic = historic.dropna().reset_index(drop=True)

    # --- 3. Configuración de capital y comisiones ---
    cash = Config.initial_capital if initial_cash is None else initial_cash
    COM = Config.COM
    Borrow_Rate = Config.BRate
    stop_loss = Config.sl
    take_profit = Config.tp
    capital_pct_exp = Config.cap_exp

    # --- 4. Inicializar tracking ---
    active_long_positions, active_short_positions, port_value = [], [], [cash]
    closed_positions = []

    # --- 5. Loop de backtest ---
    for i, row in historic.iterrows():
        price = row.Close
        n_shares = (cash * capital_pct_exp) / price

        # --- Cerrar posiciones LONG ---
        for pos in active_long_positions.copy():
            if price >= pos.tp or price <= pos.sl:
                cash += price * pos.n_shares * (1 - COM)
                pos.profit = (price - pos.price) * pos.n_shares
                closed_positions.append(pos)
                active_long_positions.remove(pos)

        # --- Cerrar posiciones SHORT ---
        for pos in active_short_positions.copy():
            days_held = i - pos.open_index
            borrow_cost = (Borrow_Rate / 252) * days_held * pos.price * pos.n_shares
            pnl = (pos.price - price) * pos.n_shares - borrow_cost

            if price <= pos.tp or price >= pos.sl:
                com = price * pos.n_shares * COM
                cash += pnl - com
                pos.profit = pnl
                closed_positions.append(pos)
                active_short_positions.remove(pos)

        # --- Abrir LONG ---
        if row.buy_signal and not active_long_positions and not active_short_positions:
            if cash >= price * n_shares * (1 + COM):
                cash -= price * n_shares * (1 + COM)
                active_long_positions.append(Position(
                    price=price, n_shares=n_shares,
                    sl=price * (1 - stop_loss), tp=price * (1 + take_profit),
                    open_index=i
                ))

        # --- Abrir SHORT ---
        if row.sell_signal and not active_short_positions and not active_long_positions:
            if cash >= price * n_shares * (1 + COM):
                cash -= price * n_shares * (1 + COM)
                active_short_positions.append(Position(
                    price=price, n_shares=n_shares,
                    sl=price * (1 + stop_loss), tp=price * (1 - take_profit),
                    open_index=i
                ))

        # --- Valor del portfolio ---
        port_value.append(get_portfolio_value(
            cash, active_long_positions, active_short_positions, price, n_shares
        ))

    # --- 6. Cerrar posiciones restantes ---
    for pos in active_long_positions:
        cash += price * pos.n_shares * (1 - COM)
        pos.profit = (price - pos.price) * pos.n_shares
        closed_positions.append(pos)

    for pos in active_short_positions:
        days_held = len(historic) - pos.open_index
        borrow_cost = (Borrow_Rate / 252) * days_held * pos.price * pos.n_shares
        pnl = (pos.price - price) * pos.n_shares - borrow_cost
        com = price * pos.n_shares * COM
        cash += pnl - com
        pos.profit = pnl
        closed_positions.append(pos)

    # --- 7. Métricas ---
    port_series = pd.Series(port_value).replace(0, np.nan).dropna()
    metrics_obj = Metrics(port_series)
    final_value = port_value[-1]
    initial_value = port_value[0]
    profit = final_value - initial_value

    metrics_dict = {
        "Calmar": metrics_obj.calmar,
        "Sharpe": metrics_obj.sharpe,
        "Sortino": metrics_obj.sortino,
        "Maximum Drawdown": metrics_obj.max_drawdown,
        "Win Rate": Metrics.win_rate(closed_positions),
        "Total Return (%)": (final_value - initial_value) / initial_value * 100,
        "Profit ($)": f"${profit:,.2f}",
        "Final Capital ($)": f"${cash:,.2f}"
    }

    return port_value, metrics_dict, cash