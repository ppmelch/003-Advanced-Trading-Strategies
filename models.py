from sklearn.utils import compute_class_weight
from libraries import *
from functions import CNN_Params, MLP_Params


def register_models(models: dict, experiment_name: str = "Advanced-Trading-Strategies") -> None:
    """
    Register trained models into the MLflow Model Registry.

    Each item in `models` is logged under a separate MLflow run, and then
    registered with the name ``{model_key}_003`` (fixed suffix).

    Parameters
    ----------
    models : dict
        Mapping from model key (str) to a trained model instance.
    experiment_name : str, default="Advanced-Trading-Strategies"
        MLflow experiment name where the runs will be recorded.

    Returns
    -------
    None
        Side effects: logs and registers models in MLflow.
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
    Load the latest version of a model from MLflow Model Registry.

    Attempts TensorFlow loader first; falls back to Keras loader on failure.

    Parameters
    ----------
    model_name : str
        Registered MLflow model name (e.g., "MLP_Model_003").

    Returns
    -------
    Any
        Loaded model artifact.
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
    Load a specific version of a model from MLflow Model Registry.

    Attempts TensorFlow loader first; falls back to Keras loader on failure.

    Parameters
    ----------
    model_name : str
        Registered model name (e.g., "CNN_Model_003").
    model_version : str
        Version number as a string (e.g., "10").

    Returns
    -------
    Any
        Loaded model artifact.
    """
    uri = f"models:/{model_name}/{model_version}"
    print(f"🔍 Cargando modelo desde MLflow: {uri}")
    try:
        return mlflow.tensorflow.load_model(model_uri=uri)
    except Exception:
        return mlflow.keras.load_model(model_uri=uri)


class Model:
    """
    Define MLP and 1D-CNN architectures for classification.

    Methods
    -------
    model_MLP(input_shape, params)
        Build a dense MLP classifier.
    model_CNN(input_shape, params)
        Build a temporal 1D-CNN classifier.
    """

    @staticmethod
    def model_MLP(input_shape: int, params: MLP_Params) -> tf.keras.Model:
        """
        Build and compile a Keras MLP model.

        Parameters
        ----------
        input_shape : int
            Number of input features.
        params : MLP_Params
            Dataclass with hyperparameters (layers, units, activations, etc.).

        Returns
        -------
        tf.keras.Model
            Compiled MLP model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_shape,)))
        for _ in range(params.dense_layers):
            model.add(tf.keras.layers.Dense(
                params.dense_units, activation=params.activation))
        model.add(tf.keras.layers.Dense(params.output_units,
                  activation=params.output_activation))
        model.compile(
            optimizer=tf.keras.optimizers.get(params.optimizer),
            loss=params.loss,
            metrics=list(params.metrics)
        )
        return model

    @staticmethod
    def model_CNN(input_shape: tuple, params: CNN_Params) -> tf.keras.Model:
        """
        Build and compile a Keras 1D-CNN model.

        Parameters
        ----------
        input_shape : tuple
            Input shape for Conv1D, typically ``(timesteps, n_features)``.
        params : CNN_Params
            Dataclass with convolutional and dense hyperparameters.

        Returns
        -------
        tf.keras.Model
            Compiled 1D-CNN model.
        """
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
        model.add(tf.keras.layers.Dense(
            params.dense_units, activation=params.activation))
        model.add(tf.keras.layers.Dense(params.output_units,
                  activation=params.output_activation))
        model.compile(
            optimizer=tf.keras.optimizers.get(params.optimizer),
            loss=params.loss,
            metrics=list(params.metrics)
        )
        return model


class Training_Model:
    """
    Training utilities for MLP and CNN models with MLflow logging.

    Methods
    -------
    training_MLP(...)
        Train an MLP with class-weighting and log metrics to MLflow.
    training_CNN(...)
        Train a 1D-CNN with class-weighting and log metrics to MLflow.
    """

    @staticmethod
    def training_MLP(
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        x_val=None,
        y_val=None,
        params_list=None,
        experiment_name: str = "Advanced-Trading-Strategies"
    ):
        """
        Train an MLP and log metrics and artifacts to MLflow.

        The labels are expected in {-1, 0, 1}. Internally, labels are shifted
        to {0, 1, 2} to match SparseCategorical* losses/metrics. Class weights
        are computed using `sklearn.utils.class_weight.compute_class_weight`.

        Parameters
        ----------
        x_train : np.ndarray
            Training features of shape (N, F).
        y_train : np.ndarray | pd.Series
            Training labels in {-1, 0, 1}.
        x_test : np.ndarray, optional
            Test features for final evaluation.
        y_test : np.ndarray | pd.Series, optional
            Test labels in {-1, 0, 1}.
        x_val : np.ndarray, optional
            Validation features for early evaluation.
        y_val : np.ndarray | pd.Series, optional
            Validation labels in {-1, 0, 1}.
        params_list : list[MLP_Params] | MLP_Params | None
            One or multiple configurations to try. If a single `MLP_Params`
            is provided, it is wrapped into a list.
        experiment_name : str, default="Advanced-Trading-Strategies"
            MLflow experiment name.

        Returns
        -------
        tf.keras.Model
            The last trained MLP model.
        """
        print("\n🚀 Entrenando modelo MLP...")
        mlflow.set_experiment(experiment_name)
        if not isinstance(params_list, list):
            params_list = [params_list]

        for params in params_list:
            run_name = f"MLP_layers{params.dense_layers}_units{params.dense_units}"
            with mlflow.start_run(run_name=run_name):
                mlflow.tensorflow.autolog(log_models=False)

                y_train_ext = np.array(y_train)
                y_train_int = y_train_ext + \
                    1 if np.min(y_train_ext) < 0 else y_train_ext

                classes_ext = np.array([-1, 0, 1])
                cw_vals = compute_class_weight(
                    class_weight="balanced", classes=classes_ext, y=y_train_ext)
                class_weight = {0: float(cw_vals[0]), 1: float(
                    cw_vals[1]), 2: float(cw_vals[2])}

                model = Model.model_MLP(
                    input_shape=x_train.shape[1], params=params)

                if x_val is not None and y_val is not None:
                    y_val_ext = np.array(y_val)
                    y_val_int = y_val_ext + \
                        1 if np.min(y_val_ext) < 0 else y_val_ext

                    hist = model.fit(
                        x_train, y_train_int,
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        verbose=params.verbose,
                        validation_data=(x_val, y_val_int),
                        class_weight=class_weight
                    )

                    y_pred_train_int = model.predict(
                        x_train, verbose=0).argmax(axis=1)
                    y_pred_val_int = model.predict(
                        x_val, verbose=0).argmax(axis=1)
                    y_pred_train_ext = y_pred_train_int - 1
                    y_pred_val_ext = y_pred_val_int - 1

                    metrics_dict = {
                        "train_accuracy": float(hist.history["accuracy"][-1]),
                        "train_loss": float(hist.history["loss"][-1]),
                        "val_accuracy": float(hist.history.get("val_accuracy", [0])[-1]),
                        "val_loss": float(hist.history.get("val_loss", [0])[-1]),
                        "train_f1": float(f1_score(y_train_ext, y_pred_train_ext, average="weighted")),
                        "val_f1": float(f1_score(y_val_ext, y_pred_val_ext, average="weighted")),
                    }
                    mlflow.log_metrics(metrics_dict)

                else:
                    hist = model.fit(
                        x_train, y_train_int,
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        verbose=params.verbose,
                        validation_split=0.2,
                        class_weight=class_weight
                    )

                    split_idx = int(len(x_train) * 0.8)
                    x_val_split = x_train[split_idx:]
                    y_val_split_ext = y_train_ext[split_idx:]

                    y_pred_train_int = model.predict(
                        x_train, verbose=0).argmax(axis=1)
                    y_pred_val_int = model.predict(
                        x_val_split, verbose=0).argmax(axis=1)
                    y_pred_train_ext = y_pred_train_int - 1
                    y_pred_val_ext = y_pred_val_int - 1

                    metrics_dict = {
                        "train_accuracy": float(hist.history["accuracy"][-1]),
                        "train_loss": float(hist.history["loss"][-1]),
                        "val_accuracy": float(hist.history.get("val_accuracy", [0])[-1]),
                        "val_loss": float(hist.history.get("val_loss", [0])[-1]),
                        "train_f1": float(f1_score(y_train_ext, y_pred_train_ext, average="weighted")),
                        "val_f1": float(f1_score(y_val_split_ext, y_pred_val_ext, average="weighted")),
                    }
                    mlflow.log_metrics(metrics_dict)

                if x_test is not None and y_test is not None:
                    y_test_ext = np.array(y_test)
                    y_test_int = y_test_ext + \
                        1 if np.min(y_test_ext) < 0 else y_test_ext

                    ev = model.evaluate(x_test, y_test_int, verbose=0)
                    test_loss = float(ev[0]) if isinstance(
                        ev, (list, tuple)) else float(ev)
                    test_acc = float(ev[1]) if isinstance(
                        ev, (list, tuple)) and len(ev) > 1 else None

                    y_pred_test_int = model.predict(
                        x_test, verbose=0).argmax(axis=1)
                    y_pred_test_ext = y_pred_test_int - 1
                    test_f1 = float(
                        f1_score(y_test_ext, y_pred_test_ext, average="weighted"))

                    test_metrics = {"test_loss": test_loss, "test_f1": test_f1}
                    if test_acc is not None:
                        test_metrics["test_accuracy"] = test_acc
                    mlflow.log_metrics(test_metrics)

        return model

    @staticmethod
    def training_CNN(
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        x_val=None,
        y_val=None,
        params_list=None,
        experiment_name: str = "Advanced-Trading-Strategies"
    ):
        """
        Train a 1D-CNN and log metrics and artifacts to MLflow.

        Labels are expected in {-1, 0, 1} and internally shifted to {0,1,2}.
        Class weights are computed from the original {-1,0,1} labels.
        Input `x_train` is expected to have shape ``(N, timesteps, features)``.

        Parameters
        ----------
        x_train : np.ndarray
            Training sequences of shape (N, T, F).
        y_train : np.ndarray | pd.Series
            Training labels in {-1, 0, 1}.
        x_test : np.ndarray, optional
            Test sequences for final evaluation.
        y_test : np.ndarray | pd.Series, optional
            Test labels in {-1, 0, 1}.
        x_val : np.ndarray, optional
            Validation sequences.
        y_val : np.ndarray | pd.Series, optional
            Validation labels in {-1, 0, 1}.
        params_list : list[CNN_Params] | CNN_Params | None
            One or multiple configurations to try.
        experiment_name : str, default="Advanced-Trading-Strategies"
            MLflow experiment name.

        Returns
        -------
        tf.keras.Model
            The last trained CNN model.
        """
        print("\n🚀 Entrenando modelo CNN...")
        mlflow.set_experiment(experiment_name)
        if not isinstance(params_list, list):
            params_list = [params_list]

        for params in params_list:
            run_name = f"CNN_layers{params.conv_layers}_filters{params.filters}"
            with mlflow.start_run(run_name=run_name):
                mlflow.tensorflow.autolog(log_models=False)

                y_train_ext = np.array(y_train)
                y_train_int = y_train_ext + \
                    1 if np.min(y_train_ext) < 0 else y_train_ext

                classes_ext = np.array([-1, 0, 1])
                cw_vals = compute_class_weight(
                    class_weight="balanced", classes=classes_ext, y=y_train_ext)
                class_weight = {0: float(cw_vals[0]), 1: float(
                    cw_vals[1]), 2: float(cw_vals[2])}

                model = Model.model_CNN(
                    input_shape=x_train.shape[1:], params=params)

                if x_val is not None and y_val is not None:
                    y_val_ext = np.array(y_val)
                    y_val_int = y_val_ext + \
                        1 if np.min(y_val_ext) < 0 else y_val_ext

                    hist = model.fit(
                        x_train, y_train_int,
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        verbose=params.verbose,
                        validation_data=(x_val, y_val_int),
                        class_weight=class_weight
                    )

                    y_pred_train_int = model.predict(
                        x_train, verbose=0).argmax(axis=1)
                    y_pred_val_int = model.predict(
                        x_val, verbose=0).argmax(axis=1)
                    y_pred_train_ext = y_pred_train_int - 1
                    y_pred_val_ext = y_pred_val_int - 1

                    metrics_dict = {
                        "train_accuracy": float(hist.history["accuracy"][-1]),
                        "train_loss": float(hist.history["loss"][-1]),
                        "val_accuracy": float(hist.history.get("val_accuracy", [0])[-1]),
                        "val_loss": float(hist.history.get("val_loss", [0])[-1]),
                        "train_f1": float(f1_score(y_train_ext, y_pred_train_ext, average="weighted")),
                        "val_f1": float(f1_score(y_val_ext, y_pred_val_ext, average="weighted")),
                    }
                    mlflow.log_metrics(metrics_dict)

                else:
                    hist = model.fit(
                        x_train, y_train_int,
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        verbose=params.verbose,
                        validation_split=0.2,
                        class_weight=class_weight
                    )

                    split_idx = int(len(x_train) * 0.8)
                    x_val_split = x_train[split_idx:]
                    y_val_split_ext = y_train_ext[split_idx:]

                    y_pred_train_int = model.predict(
                        x_train, verbose=0).argmax(axis=1)
                    y_pred_val_int = model.predict(
                        x_val_split, verbose=0).argmax(axis=1)
                    y_pred_train_ext = y_pred_train_int - 1
                    y_pred_val_ext = y_pred_val_int - 1

                    metrics_dict = {
                        "train_accuracy": float(hist.history["accuracy"][-1]),
                        "train_loss": float(hist.history["loss"][-1]),
                        "val_accuracy": float(hist.history.get("val_accuracy", [0])[-1]),
                        "val_loss": float(hist.history.get("val_loss", [0])[-1]),
                        "train_f1": float(f1_score(y_train_ext, y_pred_train_ext, average="weighted")),
                        "val_f1": float(f1_score(y_val_split_ext, y_pred_val_ext, average="weighted")),
                    }
                    mlflow.log_metrics(metrics_dict)

                if x_test is not None and y_test is not None:
                    y_test_ext = np.array(y_test)
                    y_test_int = y_test_ext + \
                        1 if np.min(y_test_ext) < 0 else y_test_ext

                    ev = model.evaluate(x_test, y_test_int, verbose=0)
                    test_loss = float(ev[0]) if isinstance(
                        ev, (list, tuple)) else float(ev)
                    test_acc = float(ev[1]) if isinstance(
                        ev, (list, tuple)) and len(ev) > 1 else None

                    y_pred_test_int = model.predict(
                        x_test, verbose=0).argmax(axis=1)
                    y_pred_test_ext = y_pred_test_int - 1
                    test_f1 = float(
                        f1_score(y_test_ext, y_pred_test_ext, average="weighted"))

                    test_metrics = {"test_loss": test_loss, "test_f1": test_f1}
                    if test_acc is not None:
                        test_metrics["test_accuracy"] = test_acc
                    mlflow.log_metrics(test_metrics)

        return model
