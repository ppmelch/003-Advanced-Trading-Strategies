from libraries import *
from indicators import Indicators
from optimizer import dataset_split, clean_data


data = clean_data("AZO", "15y")
train , test , validation = dataset_split(data)




def main():
    pass

if __name__ == "__main__":
    main()
