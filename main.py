
from libraries import *
from optimizer import dataset_split
from indicators import Indicators


# Añadir una función para los Datos 
data = yf.download("AZO", start="2010-10-10", end="2025-10-10", progress=False).reset_index(drop=True)
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

train , test , validation = dataset_split(data)


def main():
    pass

if __name__ == "__main__":
    main()
