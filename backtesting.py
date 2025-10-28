from libraries import *
from functions import Position, Config, get_portfolio_value

def _resolve_signals(df: pd.DataFrame, col: str = "signal") -> pd.Series:
    """Devuelve una serie con señales (-1, 0, 1)."""
    if col in df.columns:
        s = df[col]
    elif "final_signal" in df.columns:
        s = df["final_signal"]
    else:
        s = pd.Series(0, index=df.index)
    return s.where(s.isin([-1, 0, 1]), 0).astype(int)


def backtest(
    data: pd.DataFrame,
    cash: float = Config.initial_capital,
    sl: float = Config.sl,
    tp: float = Config.tp,
    COM: float = Config.COM,
    cap_exp: float = Config.cap_exp,
    borrow_rate_annual: float = 0.0025
) -> tuple[pd.Series, float, float, int, int, int, int, list[dict], list[dict]]:
    """
    Backtest con comisiones, stop-loss, take-profit y borrow rate para cortos.
    Usa señales -1, 0, 1 para abrir/cerrar posiciones.
    """
    df = data.copy()
    signals = _resolve_signals(df)
    longs, shorts = [], []
    port_hist, wins, trades = [], 0, 0
    buy = sell = hold = 0

    # tasa diaria equivalente al borrow rate anual
    borrow_rate_daily = Config.BRate / 252

    for idx, row in df.iterrows():
        price = float(row["Close"])
        sig = int(signals.loc[idx])

        # --- Cobro de tasa de préstamo por cortos abiertos ---
        for pos in shorts:
            interest_cost = price * pos.n_shares * borrow_rate_daily
            cash -= interest_cost  # resta el costo diario al efectivo

        # --- Cerrar largos ---
        for pos in longs.copy():
            if price <= pos.sl or price >= pos.tp:
                pnl = (price - pos.price) * pos.n_shares
                cash += price * pos.n_shares * (1 - COM)
                wins += pnl >= 0
                trades += 1
                longs.remove(pos)

        # --- Cerrar cortos ---
        for pos in shorts.copy():
            if price >= pos.sl or price <= pos.tp:
                pnl = (pos.price - price) * pos.n_shares
                cash += pnl - price * pos.n_shares * COM
                wins += pnl >= 0
                trades += 1
                shorts.remove(pos)

        # --- Nuevas operaciones ---
        if sig == 1 and not longs:  # abrir largo
            n = (cash * cap_exp / price)
            cost = price * n * (1 + COM)
            if cash >= cost:
                cash -= cost
                longs.append(Position(n, price, price*(1-sl), price*(1+tp)))
                buy += 1
        elif sig == -1 and not shorts:  # abrir corto
            n = (cash * cap_exp / price)
            fee = price * n * COM
            if cash >= fee:
                cash -= fee
                shorts.append(Position(n, price, price*(1+sl), price*(1-tp)))
                sell += 1
        else:
            hold += 1


        if longs:
            n_shares = longs[-1].n_shares
        elif shorts:
            n_shares = shorts[-1].n_shares
        else:
            n_shares = 0  # sin posiciones abiertas

        # --- Valor actual del portafolio ---
        value = get_portfolio_value(cash, longs, shorts, price, n_shares)
        port_hist.append(value)

    # --- Cierre final ---
    if len(df) > 0:
        last_price = df["Close"].iloc[-1]
        for p in longs:
            cash += last_price * p.n_shares * (1 - COM)
        for p in shorts:
            # se liquida el corto y se descuenta la comisión
            cash += (p.price - last_price) * p.n_shares - last_price * p.n_shares * COM

    win_rate = wins / trades if trades > 0 else 0
    total_trades = trades
    data_drift_results = []
    p_value_results = []

    port_series = pd.Series(port_hist, index=df.index)
    return port_series, cash, win_rate, buy, sell, hold, total_trades, data_drift_results, p_value_results
