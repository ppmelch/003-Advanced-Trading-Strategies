from libraries import *
from backtesting import backtest
from optimizer import dateset_split
from models import model_MLP, model_CNN
from sklearn.preprocessing import StandardScaler
from functions import CNN_Params, MLP_Params, Params_Indicators, BacktestingCapCOM
from indicators import Indicadores  
from sklearn.neural_network import MLPClassifier

# --- 1. Descargar datos ---
data = yf.download("AZO", start="2010-10-10", end="2025-10-10", progress=False)
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

indicators = Indicadores(params=Params_Indicators())
data = indicators.get_data(data)

X = data.drop(columns=["Target"])
y = data["Target"]

X_train, X_test, X_validation = dateset_split(X, 0.6, 0.2, 0.2)
y_train, y_test, y_validation = dateset_split(y, 0.6, 0.2, 0.2)

optimization_metric = "Calmar"  # Options: 'Sharpe', 'Sortino', 'Calmar'

def main():
    # --- 3. Escalado ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_validation_scaled = scaler.transform(X_validation)

    # --- 4. Entrenamiento MLP ---
    print("\n=== Training MLP model ===")
    acc, val_acc, class_report, y_pred_mlp = model_MLP(
        X_train_scaled, X_test_scaled, X_validation_scaled,
        y_train, y_test, y_validation
    )
    print(f"MLP Test Accuracy: {acc:.4f} | Validation Accuracy: {val_acc:.4f}")
    print(class_report)

    # --- 5. Entrenamiento CNN ---
    print("\n=== Training CNN model ===")
    cnn_model, cnn_metrics, y_pred_cnn = model_CNN(
        X_train_scaled, y_train,
        lookback=CNN_Params.lookback,
        params=CNN_Params.__dict__,
        name="CNN Trading"
    )
    print(f"CNN Val Accuracy: {cnn_metrics['val_accuracy']:.4f}")



if __name__ == "__main__":
    main()
