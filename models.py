def register_models(models: dict, experiment_name="Advanced-Trading-Strategies"):
    """
    Registers trained models in MLflow Model Registry.
    Args:
        models (dict): A dictionary with model names as keys and trained model objects as values.
        experiment_name (str): The name of the MLflow experiment to use.
    Returns:
        None
    """
    mlflow.set_experiment(experiment_name)
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path=f"{name}_artifact",
                registered_model_name=f"{name}_003",
            )
            print(f"✅ Modelo '{name}' registrado en MLflow como '{name}_003'")


def _load_latest(model_name: str):
    """
    Load the latest version of a registered model from MLflow.
    Args:
        model_name (str): The name of the registered model.
    Returns:
        The loaded model.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(int(v.version) for v in versions)
    uri = f"models:/{model_name}/{latest}"
    print(f"🔍 Cargando última versión del modelo: {uri}")
    try:
        return mlflow.tensorflow.load_model(uri)
    except Exception:
        return mlflow.keras.load_model(uri)


def model_name_version(model_name: str, model_version: str):
    """
    Load a specific version of a registered model from MLflow.
    Args:
        model_name (str): The name of the registered model.
        model_version (str): The version number of the model to load.
    Returns:
        The loaded model.
    """
    uri = f"models:/{model_name}/{model_version}"
    print(f"🔍 Cargando modelo desde MLflow: {uri}")
    try:
        return mlflow.tensorflow.load_model(model_uri=uri)
    except Exception:
        return mlflow.keras.load_model(model_uri=uri)


class Model:
    """
    Define MLP and CNN model architectures used by the training utilities.

    Contains static helper methods that build and compile Keras models:
    - model_MLP(input_shape, params: MLP_Params) -> tf.keras.Model
    - model_CNN(input_shape: tuple, params: CNN_Params) -> tf.keras.Model
    """

    @staticmethod
    def model_MLP(input_shape, params: MLP_Params):
        """
        Builds and compiles a Keras MLP model based on provided hyperparameters.
        Args:
            input_shape (int): The number of input features.
            params (MLP_Params): Hyperparameters for the MLP model.
        Returns:
            tf.keras.Model: The compiled MLP model.
        """

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_shape,)),
            *[
                tf.keras.layers.Dense(params.dense_units,
                                      activation=params.activation)
                for _ in range(params.dense_layers)
            ],
            tf.keras.layers.Dense(params.output_units,
                                  activation=params.output_activation)
        ])
        model.compile(
            optimizer=params.optimizer,
            loss=params.loss,
            metrics=["accuracy"]
        )
        return model

    @staticmethod
    def model_CNN(input_shape: tuple, params: CNN_Params):
        """
        Builds and compiles a Keras CNN model based on provided hyperparameters.
        Args:
            input_shape (tuple): The shape of the input data (timesteps, features).
            params (CNN_Params): Hyperparameters for the CNN model.
        Returns:
            tf.keras.Model: The compiled CNN model.
        """

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        filters = params.filters
        for _ in range(params.conv_layers):
            model.add(tf.keras.layers.Conv1D(filters, params.kernel_size,
                      activation=params.activation, padding="same"))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            filters *= 2
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            params.dense_units, activation=params.activation))
        model.add(tf.keras.layers.Dense(params.output_units,
                  activation=params.output_activation))
        model.compile(
            optimizer=params.optimizer,
            loss=params.loss,
            metrics=["accuracy"]
        )
        return model


class Training_Model:
    """T
    Training utilities for MLP and CNN models with MLflow logging.
    Contains static methods to train MLP and CNN models while logging metrics to MLflow:
    - training_MLP(x_train, y_train, x_test=None, y_test=None,
        x_val=None, y_val=None, params_list=None) -> tf.keras.Model
    - training_CNN(x_train, y_train, x_test=None, y_test=None,
        x_val=None, y_val=None, params_list=None) -> tf.keras.Model
    """

    @staticmethod
    def _log_final_metrics(hist, y_true_train, y_pred_train, y_true_val=None, y_pred_val=None):
        """
        Record final training and validation metrics to MLflow.
        Parameters:
            hist : tf.keras.callbacks.History
            Training history object from model.fit().
            y_true_train : np.ndarray
            True labels for training data.
            y_pred_train : np.ndarray
            Predicted labels for training data.
            y_true_val : np.ndarray, optional
            True labels for validation data.
            y_pred_val : np.ndarray, optional
            Predicted labels for validation data.
        Returns:
            None
            """
        metrics = {
            "train_accuracy": hist.history["accuracy"][-1],
            "train_loss": hist.history["loss"][-1],
            "val_accuracy": hist.history.get("val_accuracy", [0])[-1],
            "val_loss": hist.history.get("val_loss", [0])[-1],
            "train_f1": f1_score(y_true_train, y_pred_train, average="weighted"),
        }

        if y_true_val is not None and y_pred_val is not None:
            metrics["val_f1"] = f1_score(
                y_true_val, y_pred_val, average="weighted")

        mlflow.log_metrics(metrics)
        print(f"📊 Métricas finales: {metrics}")

    @staticmethod
    def training_MLP(x_train, y_train, x_test=None, y_test=None, x_val=None, y_val=None, params_list=None):
        """
        Trains an MLP model with given training data and hyperparameters, logging metrics to MLflow.
        Parameters:
            x_train : np.ndarray
                Training feature data.
            y_train : np.ndarray
                Training labels.
            x_test : np.ndarray, optional
                Test feature data.
            y_test : np.ndarray, optional
                Test labels.
            x_val : np.ndarray, optional
                Validation feature data.
            y_val : np.ndarray, optional
                Validation labels.
            params_list : list of MLP_Params
                List of hyperparameter configurations to try.
        Returns:
            tf.keras.Model
                The trained MLP model.
        """

        print("\n--- Entrenando MLP ---\n")
        mlflow.tensorflow.autolog(log_models=False)

        if y_train is not None:
            y_train = y_train + 1

        input_shape = x_train.shape[1]
        mlflow.set_experiment("Advanced-Trading-Strategies")

        for p in params_list:
            run_name = f"MLP_layers{p.dense_layers}_units{p.dense_units}"
            with mlflow.start_run(run_name=run_name):
                model = Model.model_MLP(input_shape, p)

                # 🔥 Usa 20% del train para validación
                hist = model.fit(
                    x_train, y_train,
                    validation_split=0.2,
                    batch_size=p.batch_size,
                    epochs=p.epochs,
                    verbose=0
                )

                # Predicciones para métricas F1
                y_pred_train = model.predict(x_train, verbose=0).argmax(axis=1)
                y_true_train = y_train

                # Subconjunto de validación (de validation_split)
                split_idx = int(len(x_train) * 0.8)
                x_val_ = x_train[split_idx:]
                y_val_ = y_train[split_idx:]
                y_pred_val = model.predict(x_val_, verbose=0).argmax(axis=1)

                Training_Model._log_final_metrics(
                    hist,
                    y_true_train=y_true_train,
                    y_pred_train=y_pred_train,
                    y_true_val=y_val_,
                    y_pred_val=y_pred_val,
                )

        return model

    @staticmethod
    def training_CNN(x_train, y_train, x_test=None, y_test=None, x_val=None, y_val=None, params_list=None):
        """
        Trains a CNN model with given training data and hyperparameters, logging metrics to MLflow.
        Parameters:
            x_train : np.ndarray
                Training feature data.
            y_train : np.ndarray
                Training labels.
            x_test : np.ndarray, optional
                Test feature data.
            y_test : np.ndarray, optional
                Test labels.
            x_val : np.ndarray, optional
                Validation feature data.
            y_val : np.ndarray, optional
                Validation labels.
            params_list : list of CNN_Params
                List of hyperparameter configurations to try.
        Returns:
            tf.keras.Model
                The trained CNN model.
        """

        print("\n--- Entrenando CNN ---\n")
        mlflow.tensorflow.autolog(log_models=False)

        if y_train is not None:
            y_train = y_train + 1

        input_shape = x_train.shape[1:]
        mlflow.set_experiment("Advanced-Trading-Strategies")

        for p in params_list:
            run_name = f"CNN_layers{p.conv_layers}_filters{p.filters}"
            with mlflow.start_run(run_name=run_name):
                model = Model.model_CNN(input_shape, p)

                hist = model.fit(
                    x_train, y_train,
                    validation_split=0.2,
                    batch_size=p.batch_size,
                    epochs=p.epochs,
                    verbose=0
                )

                y_pred_train = model.predict(x_train, verbose=0).argmax(axis=1)
                y_true_train = y_train

                split_idx = int(len(x_train) * 0.8)
                x_val_ = x_train[split_idx:]
                y_val_ = y_train[split_idx:]
                y_pred_val = model.predict(x_val_, verbose=0).argmax(axis=1)

                Training_Model._log_final_metrics(
                    hist,
                    y_true_train=y_true_train,
                    y_pred_train=y_pred_train,
                    y_true_val=y_val_,
                    y_pred_val=y_pred_val,
                )

        return model

