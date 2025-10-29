# 🧠 003 Advanced Trading Strategies: Deep Learning  
### ITESO — Market Microstructure and Trading Systems  
**Autores:**  
- José Armando Melchor Soto  
- Rolando Fortanell Canedo  

---

## 📘 Descripción General

Este proyecto desarrolla una **estrategia de trading sistemática** basada en **modelos de Deep Learning** entrenados sobre **características técnicas de series de tiempo**.  
Se implementan y comparan múltiples arquitecturas neuronales (MLP, CNN, y opcionalmente LSTM) para predecir señales de mercado:  
- **Long (1)** → Comprar  
- **Hold (0)** → Mantener  
- **Short (-1)** → Vender  

El sistema incluye:  
- **Ingeniería de características** (momentum, volatilidad, volumen)  
- **Monitoreo de drift de datos**  
- **Seguimiento de experimentos en MLflow**  
- **Backtesting robusto** con condiciones reales de mercado  

---

## 🎯 Objetivos

1. Construir una estrategia de trading sistemática con modelos de Deep Learning.  
2. Implementar ingeniería de características para series de tiempo.  
3. Entrenar y comparar arquitecturas **MLP** y **CNN**.  
4. Registrar y rastrear experimentos con **MLflow**.  
5. Monitorear **data drift** entre conjuntos de entrenamiento, prueba y validación.  
6. Evaluar el desempeño mediante **backtesting realista** considerando comisiones, costos de préstamo y límites SL/TP.

---

## 🧩 Estructura del Proyecto

```
003-Advanced-Trading-Strategies/
│
├── data_processing.py          # Limpieza, ingeniería de características y normalización
├── functions.py                # Clases de configuración, parámetros y posiciones
├── libraries.py                # Librerías principales y configuración global
├── metrics.py                  # Cálculo de métricas de desempeño financiero
├── models.py                   # Definición, entrenamiento y registro de modelos DL
├── normalization.py            # Normalización de indicadores técnicos
├── prints.py                   # Impresión de resultados y backtesting integrado
├── main.py                     # Ejecución principal: evaluación y backtesting final
├── requirements.txt            # Dependencias del entorno
└── README.md                   # Este archivo
```

---

## ⚙️ Instalación

### 1️⃣ Crear entorno virtual

```bash
python -m venv env
source env/bin/activate     # Linux/Mac
env\Scripts\activate      # Windows
```

### 2️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3️⃣ Configurar MLflow (opcional)

Si deseas registrar experimentos en MLflow:

```bash
mlflow ui
```
Esto iniciará la interfaz en: [http://localhost:5000](http://localhost:5000)

---

## 💾 Datos

El proyecto descarga automáticamente **15 años de datos diarios** de `AAPL` desde **Yahoo Finance** usando `yfinance`.  
No se requieren archivos externos de datos.

```python
from data_processing import clean_data
data = clean_data("AAPL", "15y")
```

Los datos se dividen cronológicamente:
- **60% Train**
- **20% Test**
- **20% Validation**

---

## 🧠 Modelos de Deep Learning

### 🟢 MLP — Multilayer Perceptron
- Densas: 2 capas ocultas de 128 neuronas  
- Activación: ReLU  
- Salida: 3 neuronas Softmax  
- Loss: Sparse Categorical Crossentropy  
- Epochs: 100  
- Batch size: 252  

### 🔵 CNN — Convolutional Neural Network
- Lookback: 20 pasos
- Capas convolucionales: 2  
- Filtros: 64, Kernel: 3  
- Pooling: MaxPooling1D  
- Dense head: 64 unidades  
- Epochs: 60  
- Batch size: 252  

---

## 📊 Backtesting

El backtesting simula operaciones con:
- **Capital inicial:** $1,000,000  
- **Comisión por operación:** 0.125%  
- **Borrow rate (shorts):** 0.25% anual  
- **SL:** 2%  
- **TP:** 5%  
- **Capital expuesto por operación:** 30%  

Resultados por split (Train, Test, Val) incluyen:
- **Sharpe Ratio**
- **Sortino Ratio**
- **Calmar Ratio**
- **Max Drawdown**
- **Win Rate**
- **Gráficas de crecimiento del portafolio**

---

## 🧪 Ejecución del Proyecto

### 🔹 Entrenamiento (si se requiere)
Entrenar y registrar modelos:
```python
from models import Training_Model, Model, MLP_Params, CNN_Params
# Entrenar un MLP
model = Training_Model.training_MLP(x_train, y_train, x_val=x_val, y_val=y_val, params_list=MLP_Params())
```

### 🔹 Evaluación y Backtesting
El archivo `main.py` evalúa modelos registrados en MLflow y genera backtests automáticos:

```bash
python main.py
```

Salidas:
- Métricas de desempeño por split
- Curvas de portafolio (Train, Test, Val)
- Resumen final de rendimiento

---

## 📈 Métricas Clave

| Métrica | Descripción |
|----------|--------------|
| **Sharpe Ratio** | Rentabilidad ajustada por riesgo total |
| **Sortino Ratio** | Rentabilidad ajustada por riesgo a la baja |
| **Max Drawdown** | Máxima pérdida desde un pico |
| **Calmar Ratio** | Retorno anual / Drawdown máximo |
| **Win Rate** | Porcentaje de operaciones ganadoras |

---

## 🧮 MLflow Tracking

Cada modelo se ejecuta en un **experimento MLflow**:
- Nombre: `Advanced-Trading-Strategies`
- Se registran:
  - Hiperparámetros
  - Métricas de train/val/test
  - F1-score ponderado
  - Curvas de aprendizaje
- Modelos registrados como:
  - `MLP_Model_003`
  - `CNN_Model_003`

Ejemplo de carga desde MLflow:
```python
from models import model_name_version
mlp = model_name_version("MLP_Model_003", "7")
```

---

## 🔍 Resultados Finales

| Modelo | Split | Final Portfolio | Sharpe | Sortino | Calmar | MDD | Win Rate |
|---------|--------|----------------|--------|----------|---------|-----|-----------|
| **MLP** | Train | $1.44M | + | + | + | ↓12% | 57% |
| **MLP** | Test | $1.19M | + | + | + | ↓15% | 55% |
| **CNN** | Train/Test | ↓50% | - | - | - | ↑58% | <30% |

✅ **Conclusión:**  
El modelo **MLP** fue más estable y rentable, mientras que la **CNN** no logró generalizar adecuadamente.

---

## 🧾 Bibliografía

- Sharpe, W. F. (1966). *Mutual fund performance.* *Journal of Business.*  
- Sortino, F. A., & Price, L. N. (1994). *Performance measurement in a downside risk framework.*  
- Chan, E. (2009). *Quantitative Trading: How to Build Your Own Algorithmic Trading Business.* Wiley.  
- OpenAI. (2025). *ChatGPT (GPT-5-mini).* https://openai.com/chatgpt

---

## 🛠️ Tecnologías

- **Python 3.11+**  
- **TensorFlow / Keras**  
- **scikit-learn**  
- **pandas / numpy / matplotlib / seaborn**  
- **ta (Technical Analysis Library)**  
- **MLflow**  
- **yfinance**
