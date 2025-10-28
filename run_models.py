from libraries import *
from prints import results
from models import Training_Model
from functions import CNN_Params, MLP_Params
from models import  model_name_version, register_models
from data_processing import dataset_split, clean_data, process_dataset, target

def Base():
    # ============== DATA ==============
    data = clean_data("AZO", "15y")
    train, test, val = dataset_split(data)

    train_data, train_ind, train_price = process_dataset(train)
    test_data,  test_ind,  test_price  = process_dataset(test)
    val_data,   val_ind,   val_price   = process_dataset(val)

    # Targets en -1/0/1 (mantén así en tu pipeline)
    x_train, y_train = target(train_price)
    x_test,  y_test  = target(test_price)
    x_val,   y_val   = target(val_price)

    # Shift SOLO para entrenar Keras (0/1/2), X a float32
    x_train = np.asarray(x_train, dtype="float32")
    x_test  = np.asarray(x_test,  dtype="float32")
    x_val   = np.asarray(x_val,   dtype="float32")
   
    # Datasets p/ MLP (2D) — copiamos DF para no mutar
    datasets_mlp = {
        "train": (train_data.copy(), x_train),
        "test":  (test_data.copy(),  x_test),
        "val":   (val_data.copy(),   x_val),
    }

    # Datasets p/ CNN (3D: (timesteps, 1))
    x_train_c = x_train[..., None]
    x_test_c  = x_test[..., None]
    x_val_c   = x_val[..., None]
    datasets_cnn = {
        "train": (train_data.copy(), x_train_c),
        "test":  (test_data.copy(),  x_test_c),
        "val":   (val_data.copy(),   x_val_c),
    }

    # ============== TRAIN ==============
    tf.keras.backend.clear_session()

    mlp_params = MLP_Params()
    cnn_params = CNN_Params()

    print("\n--- Training MLP models ---")
    model_MLP = Training_Model.training_MLP(
        x_train, y_train, x_test, y_test, x_val, y_val, [mlp_params]
    )

    print("\n--- Training CNN models ---")
    model_CNN = Training_Model.training_CNN(
        x_train_c, y_train, x_test_c, y_test, x_val_c, y_val, [cnn_params]
    )

    # ============== REGISTER ==============
    models_to_register = {
        "MLP_Model": model_MLP,
        "CNN_Model": model_CNN,
    }
    register_models(models_to_register)  # tu función ya pone sufijos tipo *_003

    # ============== LOAD LATEST ==============
    from mlflow.tracking import MlflowClient
    import mlflow

    def _load_latest(model_name: str):
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest = max(int(v.version) for v in versions)
        uri = f"models:/{model_name}/{latest}"
        print(f"🔍 Loading latest model: {uri}")
        try:
            return mlflow.tensorflow.load_model(uri)
        except Exception:
            return mlflow.keras.load_model(uri)

    # Usa los nombres reales con los que quedaron registrados (p.ej. *_003)
    mlp_loaded = _load_latest("MLP_Model_003")
    cnn_loaded = _load_latest("CNN_Model_003")

    # ============== RESULTS / BACKTEST ==============
    print("\n================== RESULTS FOR MODEL: MLP ==================")
    results(datasets_mlp, mlp_loaded)   # tu results ya mantiene cash de test→val

    print("\n================== RESULTS FOR MODEL: CNN ==================")
    results(datasets_cnn, cnn_loaded)


if __name__ == "__main__":
    Base()
