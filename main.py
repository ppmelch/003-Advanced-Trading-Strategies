from libraries import *
from prints import results
from models import  model_name_version
from data_processing import dataset_split, clean_data, process_dataset, target

def main():
    data = clean_data("AZO", "15y")
    train, test, val = dataset_split(data)

    train_data, train_ind, train_price = process_dataset(train)
    test_data,  test_ind,  test_price  = process_dataset(test)
    val_data,   val_ind,   val_price   = process_dataset(val)

    y_train = target(train_price)
    y_test  = target(test_price)
    y_val   = target(val_price)


    datasets = {
        "train": (train_data, train_ind),
        "test":  (test_data,  test_ind),
        "val":   (val_data,   val_ind),
    }

    for name, (df, X) in datasets.items():
        assert hasattr(df, "reset_index"), f"{name}: data debe ser DataFrame"
        assert len(df) == len(X), f"{name}: |data| != |X| ({len(df)} vs {len(X)})"


    model_MLP = model_name_version("MLP_Model_003", "13")
    print("\n=== Results for MLP Model ===\n")
    results(datasets, model_MLP)

    model_CNN = model_name_version("CNN_Model_003", "13")
    print("\n=== Results for CNN Model ===\n")
    results(datasets, model_CNN)

if __name__ == "__main__":
    main()
