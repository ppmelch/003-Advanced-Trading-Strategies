# 003 Advanced Trading Strategies: Deep Learning  
### ITESO — Market Microstructure and Trading Systems  

**Authors:**  
- José Armando Melchor Soto  
- Rolando Fortanell Canedo  

---

## Project Overview

This project develops a **systematic trading strategy** based on **Deep Learning models** trained on **technical features from financial time series**.  
Multiple neural network architectures are implemented and compared (MLP, CNN, and optionally LSTM) to predict market signals:

- **Long (1)** → Buy  
- **Hold (0)** → Hold  
- **Short (-1)** → Sell  

The system includes:
- Feature engineering (momentum, volatility, volume)
- Data drift monitoring
- Experiment tracking with MLflow
- Robust backtesting under realistic market conditions

---

## Objectives

1. Build a systematic trading strategy using Deep Learning models.  
2. Implement feature engineering for financial time series.  
3. Train and compare MLP and CNN architectures.  
4. Track and register experiments using MLflow.  
5. Monitor data drift across training, testing, and validation datasets.  
6. Evaluate performance through realistic backtesting, including commissions, borrowing costs, and SL/TP constraints.

---

## Project Structure


```
003-Advanced-Trading-Strategies/
│
├── data_processing.py          
├── functions.py                
├── libraries.py                
├── metrics.py                  
├── models.py                   
├── normalization.py            
├── prints.py                   
├── main.py                     
├── requirements.txt            
└── README.md                   
```

---

## Installation

### Create a virtual environment

```bash
python -m venv env
source env/bin/activate     # Linux/Mac
env\Scripts\activate      # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### MLflow setup (optional)


```bash
mlflow ui
```
MLflow UI will be available at: [http://localhost:5000](http://localhost:5000)

---

Data

The project automatically downloads 15 years of daily market data for ´AAPL´ from Yahoo Finance using ´yfinance´.

```python
from data_processing import clean_data
data = clean_data("AAPL", "15y")
```

Dataset split:
- **60% Train**
- **20% Test**
- **20% Validation**

---

## Deep Learning Models 


### MLP — Multilayer Perceptron

- 2 hidden layers (128 neurons)

- ReLU activation

- Softmax output (3 classes)

- Epochs: 100

- Batch size: 252

### CNN — Convolutional Neural Network

- Lookback window: 20

- 2 convolutional layers

- 64 filters, kernel size 3

- MaxPooling1D

- Dense head (64 units)

- Epochs: 60

- Batch size: 252



---

## Backtesting Framework

- Initial capital: $1,000,000

- Transaction cost: 0.125%

- Borrow rate (short positions): 0.25% annually

- Stop Loss: 2%

- Take Profit: 5%

- Capital allocation per trade: 30%

### Metrics:

- Sharpe Ratio

- Sortino Ratio

- Calmar Ratio

- Maximum Drawdown

- Win Rate

---

## Execution

### Model Training

Entrenar y registrar modelos:
```python
from models import Training_Model, Model, MLP_Params, CNN_Params
# Traning MLP
model = Training_Model.training_MLP(x_train, y_train, x_val=x_val, y_val=y_val, params_list=MLP_Params())
```

### Evaluation and Backtesting
The `main.py` file evaluates models registered in MLflow and generates automated backtests.

```bash
python main.py
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted return using total volatility |
| **Sortino Ratio** | Risk-adjusted return using downside risk |
| **Max Drawdown** | Maximum loss from a portfolio peak |
| **Calmar Ratio** | Annualized return divided by maximum drawdown |
| **Win Rate** | Percentage of profitable trades |

---

## MLflow Tracking

Each model is executed within an **MLflow experiment**:
- Experiment name: `Advanced-Trading-Strategies`
- Logged information includes:
  - Hyperparameters
  - Train / validation / test metrics
  - Weighted F1-score
  - Learning curves
- Registered models:
  - `MLP_Model_003`
  - `CNN_Model_003`

Example of loading a model from MLflow:

```python
from models import model_name_version
mlp = model_name_version("MLP_Model_003", "7")
```

---

## Final Results

| Model | Split | Final Portfolio | Sharpe | Sortino | Calmar | Max DD | Win Rate |
|-------|--------|----------------|--------|----------|---------|---------|----------|
| **MLP** | Train | $1.44M | + | + | + | -12% | 57% |
| **MLP** | Test | $1.19M | + | + | + | -15% | 55% |
| **CNN** | Train/Test | $0.50M | - | - | - | -58% | <30% |

**Conclusion:**  
The **MLP model** was more stable and profitable, while the **CNN** failed to generalize adequately.

---

## References

- Sharpe, W. F. (1966). *Mutual fund performance.* *Journal of Business.*  
- Sortino, F. A., & Price, L. N. (1994). *Performance measurement in a downside risk framework.*  
- Chan, E. (2009). *Quantitative Trading: How to Build Your Own Algorithmic Trading Business.* Wiley.  
- OpenAI. (2025). *ChatGPT (GPT-5-mini).* https://openai.com/chatgpt

---

## Technologies

- **Python 3.11+**  
- **TensorFlow / Keras**  
- **scikit-learn**  
- **pandas / numpy / matplotlib / seaborn**  
- **ta (Technical Analysis Library)**  
- **MLflow**  
- **yfinance**
