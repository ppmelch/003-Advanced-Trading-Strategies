# --- Standard library ---
import os
import warnings
import re, datetime as dt
from dataclasses import dataclass

warnings.filterwarnings("ignore")        
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- Third-party libraries: Data analysis ---
import ta
import numpy as np
import pandas as pd
import scipy as sp
import yfinance as yf
from scipy.stats import ks_2samp
from ta.momentum import RSIIndicator, WilliamsRIndicator, ROCIndicator, AwesomeOscillatorIndicator, StochasticOscillator
from ta.volatility import BollingerBands, DonchianChannel, KeltnerChannel, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumePriceTrendIndicator, AccDistIndexIndicator

# --- Third-party libraries: Visualization ---
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import relativedelta
from IPython.display import display

# --- Third-party libraries: Machine Learning / Optimization ---
import mlflow
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score, classification_report

# --- Type hints ---
from typing import List

np.random.seed(42)

plt.rcParams['figure.facecolor'] = 'lightgrey'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'


colors = ["cornflowerblue", "indianred", "darkseagreen", "plum", "dimgray"]