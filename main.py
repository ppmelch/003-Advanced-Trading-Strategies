from libraries import *
from optimizer import dataset_split, clean_data , all_indicators


data = clean_data("AZO", "15y")
train , test , validation = dataset_split(data)

train_data = all_indicators(train)


def main():
    pass

if __name__ == "__main__":
    main()
