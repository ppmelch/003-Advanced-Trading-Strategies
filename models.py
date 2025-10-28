from libraries import *
from functions import MLP_Params, CNN_Params
from sklearn.metrics import f1_score, accuracy_score


def register_models(models: dict, experiment_name="Advanced-Trading-Strategies"):
    """
    Registra múltiples modelos TensorFlow en MLflow bajo un experimento.

    Parameters
    ----------
    models : dict
        Diccionario con formato {"run_name": model_object}
        Ejemplo: {"MLP_Run": model_MLP, "CNN_Run": model_CNN}
    experiment_name : str, optional
        Nombre del experimento MLflow. Por defecto "Advanced-Trading-Strategies".

    Notes
    -----
    - Usa mlflow.tensorflow.log_model() para cada modelo.
    - Cada modelo se guarda con su run_name y se registra bajo el mismo nombre.
    - Devuelve un dict con los run_ids de cada registro.
    """

    mlflow.set_experiment(experiment_name)
    run_ids = {}

    for run_name, model in models.items():
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path=f"Model_{run_name}",
                registered_model_name=f"{run_name}_003"
            )
            print(f"✅ Modelo '{run_name}' registrado en MLflow como '{run_name}_003'")
            run_ids[run_name] = run.info.run_id

    return run_ids

def model_name_version(model_name: str, model_version: str):
    """
    Carga un modelo desde el MLflow Model Registry dado su nombre y versión.
    Compatible con modelos registrados mediante mlflow.tensorflow.log_model().
    """
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"🔍 Cargando modelo desde MLflow: {model_uri}")

    # Usa el loader correcto según el formato registrado
    try:
        model = mlflow.tensorflow.load_model(model_uri=model_uri)
    except Exception:
        model = mlflow.keras.load_model(model_uri=model_uri)

    print(model.summary())
    return model

class Model:
    @staticmethod
    def model_MLP(input_shape, params: MLP_Params):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_shape,)))
        for _ in range(params.dense_layers):
            model.add(tf.keras.layers.Dense(params.dense_units, activation=params.activation))
        model.add(tf.keras.layers.Dense(params.output_units, activation=params.output_activation))
        model.compile(
            optimizer=tf.keras.optimizers.get(params.optimizer),
            loss=params.loss,
            metrics=list(params.metrics)
        )
        return model

    @staticmethod
    def model_CNN(input_shape: tuple, params: CNN_Params) -> tf.keras.Model:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        filters = params.filters
        for _ in range(params.conv_layers):
            model.add(tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=params.kernel_size,
                activation=params.activation,
                padding='same'
            ))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            filters *= 2
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(params.dense_units, activation=params.activation))
        model.add(tf.keras.layers.Dense(params.output_units, activation=params.output_activation))
        model.compile(
            optimizer=tf.keras.optimizers.get(params.optimizer),
            loss=params.loss,
            metrics=list(params.metrics)
        )
        return model

class Training_Model:

    @staticmethod
    def training_MLP(x_train, y_train, x_test, y_test, x_val, y_val, params_list: list[MLP_Params]) -> None:
        
        print("\n--- Training MLP models ---\n")
        y_train = y_train + 1
        y_test  = y_test  + 1
        y_val   = y_val   + 1
        input_shape = x_train.shape[1]

        mlflow.set_experiment("Advanced-Trading-Strategies")

        for p in params_list:
            run_name = f"dense{p.dense_layers}_units{p.dense_units}_activation{p.activation}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("MLP_run", run_name)
                print(f"-- Training & Running : MLP Model --")
                model = Model.model_MLP(input_shape, p)
                hist = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=p.batch_size,
                    epochs=p.epochs,
                    verbose=p.verbose
                )
                y_pred_train = model.predict(x_train, verbose=0)
                y_pred_val   = model.predict(x_val,   verbose=0)
                y_pred_test  = model.predict(x_test,  verbose=0)
                train_f1 = f1_score(y_train, y_pred_train.argmax(axis=1), average=p.Average)
                val_f1   = f1_score(y_val,   y_pred_val.argmax(axis=1),   average=p.Average)
                test_f1  = f1_score(y_test,  y_pred_test.argmax(axis=1),  average=p.Average)
                val_accuracy  = accuracy_score(y_val,  y_pred_val.argmax(axis=1))
                test_accuracy = accuracy_score(y_test, y_pred_test.argmax(axis=1))
                mlflow.log_metrics({
                    "train_f1_score": train_f1,
                    "val_f1_score":   val_f1,
                    "test_f1_score":  test_f1,
                    "val_accuracy":   val_accuracy,
                    "test_accuracy":  test_accuracy,
                })
                final_metrics = {
                    "last_accuracy":      hist.history["accuracy"][-1],
                    "last_val_accuracy":  hist.history["val_accuracy"][-1],
                    "last_loss":          hist.history["loss"][-1],
                    "last_val_loss":      hist.history["val_loss"][-1],
                }
                mlflow.log_metrics(final_metrics)
                print(final_metrics)
        return model

    @staticmethod
    def training_CNN(
        x_train: np.ndarray, y_train: np.ndarray,
        x_test: np.ndarray,  y_test:  np.ndarray,
        x_val: np.ndarray,   y_val:   np.ndarray,
        params_list: list[CNN_Params]
    ) -> None:
        print("\n--- Training CNN models ---\n")
        y_train = y_train + 1
        y_test  = y_test  + 1
        y_val   = y_val   + 1
        input_shape = x_train.shape[1:]

        mlflow.set_experiment("Advanced-Trading-Strategies")

        for p in params_list:
            run_name = f"conv{p.conv_layers}_filters{p.filters}_dense{p.dense_units}_activation{p.activation}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("CNN_run", run_name)
                print(f"-- Training & Running : CNN Model --")
                model = Model.model_CNN(input_shape, p)
                hist = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=p.batch_size,
                    epochs=p.epochs,
                    verbose=p.verbose
                )
                y_pred_train = model.predict(x_train, verbose=0)
                y_pred_val   = model.predict(x_val,   verbose=0)
                y_pred_test  = model.predict(x_test,  verbose=0)
                train_f1 = f1_score(y_train, y_pred_train.argmax(axis=1), average=p.Average)
                val_f1   = f1_score(y_val,   y_pred_val.argmax(axis=1),   average=p.Average)
                test_f1  = f1_score(y_test,  y_pred_test.argmax(axis=1),  average=p.Average)
                val_accuracy  = accuracy_score(y_val,  y_pred_val.argmax(axis=1))
                test_accuracy = accuracy_score(y_test, y_pred_test.argmax(axis=1))
                mlflow.log_metrics({
                    "train_f1_score": train_f1,
                    "val_f1_score":   val_f1,
                    "test_f1_score":  test_f1,
                    "val_accuracy":   val_accuracy,
                    "test_accuracy":  test_accuracy,
                })
                final_metrics = {
                    "last_accuracy":      hist.history["accuracy"][-1],
                    "last_val_accuracy":  hist.history["val_accuracy"][-1],
                    "last_loss":          hist.history["loss"][-1],
                    "last_val_loss":      hist.history["val_loss"][-1],
                }
                mlflow.log_metrics(final_metrics)
                print(final_metrics)
        return model


