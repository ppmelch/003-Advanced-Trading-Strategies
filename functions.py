from libraries import *

@dataclass
class Position:
    """
    Represents a position in the portfolio.

    Attributes:
    n_shares : float
        Number of shares or units in the position.
    price : float
        Entry price of the position.
    sl : float
        Stop loss price.
    tp : float
        Take profit price.
    profit : float, optional
        Realized profit or loss of the position (default: None).
    exit_price : float, optional
        Exit price of the position (default: None).
    """
    n_shares: float
    price: float
    sl: float 
    tp: float 
    profit: float = None
    exit_price: float = None


@dataclass
class Config:
    """
    Backtesting configuration with initial capital and commission.

    Attributes:
    initial_capital : float
        Initial capital for backtesting (default: 1_000_000).
    COM : float
        Commission per trade in percentage (default: 0.125 / 100).
    borrow_Rate : float6
        Borrowing rate for short positions (default: 0.25 / 100).
    """
    initial_capital: float = 1_000_000
    COM: float = 0.125 / 100
    BRate: float = 0.25 / 100
    sl : float = 0.02
    tp : float = 0.05
    cap_exp: float = 0.3

    
@dataclass
class Params_Indicators:
    """
    Parameters for technical indicators.
    Attributes:
        Momentum indicators, Volatility indicators, Volume indicators
    """
    # --- Momentum Indicators (8) ---
    rsi_7_window: int = 7 
    rsi_10_window: int = 10 
    rsi_14_window: int = 14 
    rsi_20_window: int = 20 
    awe_window1: int = 5  
    awe_window2: int = 34 
    williams_r_lbp: int = 14 
    roc_window: int = 12 
    stoch_osc_window : int = 14 
    stoch_osc_smooth : int = 3 

    # --- Volatility Indicators (8) ---
    atr_window: int = 14 
    bollinger_window: int = 20 
    bollinger_dev: int = 2 
    donchian_window: int = 20  
    keltner_window: int = 20
    keltner_atr: int = 10

    # --- Volume Indicators (4) ---
    cmf_window : int = 14 

@dataclass
class MLP_Params:
    """Hyperparameters for a Keras-based MLP model."""
    dense_layers: int = 2           
    dense_units: int = 64            
    activation: str = "relu"        
    optimizer: str = "adam"          
    output_units: int = 3            
    output_activation: str = "softmax" 
    loss: str = "sparse_categorical_crossentropy" 
    metrics: tuple = ("accuracy",)   
    batch_size: int = 32
    epochs: int = 100
    verbose : int = 2
    Average : str = 'weighted'

@dataclass
class CNN_Params:
    """Hyperparameters for a Keras-based CNN model."""
    lookback: int = 20
    conv_layers: int = 2
    filters: int = 32
    kernel_size: int = 3
    dense_units: int = 64
    activation: str = 'relu'
    optimizer: str = 'adam'
    output_units: int = 3            
    output_activation: str = "softmax" 
    epochs: int = 100
    batch_size: int = 32
    metrics: tuple = ("accuracy",)  
    verbose : int = 2
    Average : str = 'weighted'
    loss : str = "sparse_categorical_crossentropy"


def get_portfolio_value(cash: float, long_ops: list[Position], short_ops: list[Position], current_price: float, n_shares: float) -> float:
    """
    Calculates the total portfolio value at a given moment.

    Parameters:
    cash : float
        Cash available in the portfolio.
    long_ops : list[Position]
        List of active long positions.
    short_ops : list[Position]
        List of active short positions.
    current_price : float
        Current price of the asset.
    n_shares : float
        Number of shares per position.

    Returns:
    float
        Total portfolio value including long and short positions.
    """
    value = cash
    for pos in long_ops:
        value += current_price * pos.n_shares
    for pos in short_ops:
        value +=  (pos.price - current_price) * pos.n_shares
    return value
