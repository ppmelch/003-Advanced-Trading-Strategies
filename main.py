from libraries import *

# --- 1. Descargar datos ---
data = yf.download("AZO", start="2010-10-10", end="2025-10-10", progress=False).reset_index(drop=True)



def main():
    pass

if __name__ == "__main__":
    main()
