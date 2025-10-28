from libraries import *
from functions import Position, Config, get_portfolio_value

def backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    cash: float = Config.initial_capital,
    sl: float = Config.sl,
    tp: float = Config.tp,
    COM: float = Config.COM,
    cap_exp: float = Config.cap_exp
) -> tuple[pd.Series, float, float, int, int, int, int]:
    """
    Run a long/short backtest with SL/TP, commissions, and borrow costs.

    The strategy interprets signals as:
    -  1: open/maintain a single long position (if none is open)
    - -1: open/maintain a single short position (if none is open)
    -  0: hold (no new positions)
    Only one long or one short can be open at a time. Positions close on SL/TP.
    Short positions accrue a daily borrow fee using `Config.BRate / 252`.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed OHLCV data. Must include 'Close'.
    signals : pd.Series
        Series aligned (or alignable) to `data.index` with values in {-1, 0, 1}.
    cash : float, default=Config.initial_capital
        Starting cash.
    sl : float, default=Config.sl
        Stop-loss distance as a fraction of entry price (e.g., 0.02 = 2%).
    tp : float, default=Config.tp
        Take-profit distance as a fraction of entry price (e.g., 0.05 = 5%).
    COM : float, default=Config.COM
        Commission rate per trade (fraction).
    cap_exp : float, default=Config.cap_exp
        Fraction of current cash deployed per new position.

    Returns
    -------
    tuple
        (
            port_series : pd.Series
                Portfolio value over time (same index as `data`).
            cash : float
                Final cash at the end of the simulation.
            win_rate : float
                Wins / total closed trades in [0, 1].
            buy : int
                Number of long entries.
            sell : int
                Number of short entries.
            hold : int
                Number of hold steps (no new position opened).
            trades : int
                Number of closed trades.
        )
    """
    df = data.copy()
    signals = signals.reindex(df.index).fillna(0).astype(int)

    port_hist = [cash]
    longs, shorts = [], []
    wins = trades = 0
    buy = sell = hold = 0
    borrow_rate_daily = Config.BRate / 252

    for idx, row in df.iloc[1:].iterrows():
        price = float(row["Close"])
        sig = int(signals.loc[idx])

        for pos in shorts:
            cash -= price * pos.n_shares * borrow_rate_daily

        for pos in longs.copy():
            if price <= pos.sl or price >= pos.tp:
                pnl = (price - pos.price) * pos.n_shares
                cash += price * pos.n_shares * (1 - COM)
                wins += pnl >= 0
                trades += 1
                longs.remove(pos)

        for pos in shorts.copy():
            if price >= pos.sl or price <= pos.tp:
                pnl = (pos.price - price) * pos.n_shares
                cash += pnl - price * pos.n_shares * COM
                wins += pnl >= 0
                trades += 1
                shorts.remove(pos)

        if sig == 1 and not longs:
            n = cash * cap_exp / price
            cost = price * n * (1 + COM)
            if cash >= cost:
                cash -= cost
                longs.append(Position(n, price, price *
                             (1 - sl), price * (1 + tp)))
                buy += 1
        elif sig == -1 and not shorts:
            n = cash * cap_exp / price
            fee = price * n * COM
            if cash >= fee:
                cash -= fee
                shorts.append(Position(n, price, price *
                              (1 + sl), price * (1 - tp)))
                sell += 1
        else:
            hold += 1

        if longs:
            n_shares = longs[-1].n_shares
        elif shorts:
            n_shares = shorts[-1].n_shares
        else:
            n_shares = 0

        value = get_portfolio_value(cash, longs, shorts, price, n_shares)
        port_hist.append(value)

    win_rate = wins / trades if trades > 0 else 0.0
    port_series = pd.Series(port_hist, index=df.index)
    return port_series, cash, win_rate, buy, sell, hold, trades
