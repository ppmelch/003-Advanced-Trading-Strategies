from libraries import *
from models import model_MLP, model_CNN
from functions import CNN_Params , MLP_Params
from data_processing import dataset_split, clean_data, process_dataset, target


def main():
    data = clean_data("AZO", "15y")
    train, test, validation = dataset_split(data)

    train_data, train_data_indicators, train_data_price = process_dataset(train)
    x_train_p, y_train_p = target(train_data_price)

    test_data, test_data_indicators, test_data_price = process_dataset(test)
    validation_data, validation_data_indicators, validation_data_price = process_dataset(validation)

    data_combined = pd.concat([test_data , validation_data]).sort_index()
    data_combined_indicators = pd.concat([test_data_indicators , validation_data_indicators]).sort_index()
    data_combined_price = pd.concat([test_data_price , validation_data_price]).sort_index()

    # Separar target variable for combined ---
    x_train , y_train = target(data_combined_price)
    x_test , y_test = target(test_data_price)
    x_validation , y_val = target(validation_data_price)

    # -- Model MLP --
    




    # -- Model CNN --

if __name__ == "__main__":
    main()
