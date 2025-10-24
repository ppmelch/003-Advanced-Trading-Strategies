from libraries import *
from optimizer import dataset_split, clean_data , all_indicators, get_signals


def main():
    # --- Cargar datos ---
    data = clean_data("AZO", "15y")
    # --- Dividir dataset ---
    train , test , validation = dataset_split(data)
    # --- Indicadores técnicos ---
    train_data = all_indicators(train)
    # --- Signals ---
    train_data = get_signals(train_data)
    


if __name__ == "__main__":
    main()
