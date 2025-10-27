from libraries import *
from functions import MLP_Params, CNN_Params
from sklearn.metrics import f1_score, accuracy_score

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
        for p in params_list:
            run_name = f"dense{p.dense_layers}_units{p.dense_units}_activation{p.activation}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("MLP_run", run_name)
                print(f"-- Training & Running : {run_name} --")
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
        for p in params_list:
            run_name = f"conv{p.conv_layers}_filters{p.filters}_dense{p.dense_units}_activation{p.activation}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("CNN_run", run_name)
                print(f"-- Training & Running : {run_name} --")
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


