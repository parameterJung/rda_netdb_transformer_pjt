from keras.layers import Dense, SimpleRNN, LSTM
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mlflow
from tensorflow.python.keras import layers, models
from keras.layers import*

from bson.binary import Binary
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling1D, Input, LayerNormalization, LSTM, MultiHeadAttention, ReLU, SimpleRNN
from keras.models import load_model, Model, model_from_json, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

class MongoDatabase:
    # Initializer method, called when a new instance of MongoDatabase is created
    def __init__(self):
        # Connection string for MongoDB
        CONNECTION_STRING = "mongodb://netdb:netdb3230!@10.255.93.173:27017/"
        # Creating MongoClient object using the connection string
        self.client = MongoClient(CONNECTION_STRING)

    def _fetch_data(self, collection_name, limit=None):
        """Private method to fetch data from a specified collection in MongoDB."""
        try:
            collection = self.client["TestAPI"][collection_name]
            cursor = collection.find({}).limit(limit) if limit else collection.find({})
            return pd.DataFrame(list(cursor))
        except Exception as e:
            print(f"Error while fetching data from {collection_name}: {e}")
            return None

    def get_environment(self, limit=None):
        """Public method to fetch environment data from the 'GH2' collection."""
        return self._fetch_data("GH1", limit)

    def get_growth(self, limit=None):
        """Public method to fetch growth data from the 'hydroponics_length2' collection."""
        return self._fetch_data("hydroponics_length1", limit)


def create_dataset(X, y, look_back=1):
    """
    Create dataset for time-series forecasting.

    Parameters:
    - X: Input time-series data (features).
    - y: Output time-series data (target).
    - look_back (default=1): Number of previous time steps to use as input variables
                             to predict the next time step.

    Returns:
    - dataX: List of the input sequences.
    - dataY: List of the output sequences.
    """

    dataX, dataY = [], []  # Initialize empty lists to hold our transformed sequences.

    # For each possible sequence in the input data...
    for i in range(len(X) - look_back):
        # Extract a sequence of 'look_back' features from the input data.
        sequence = X[i:(i + look_back), :]
        dataX.append(sequence)

        # Extract the output for this sequence from the 'y' data.
        output = y[i + look_back]
        dataY.append(output)

    # Convert the lists into NumPy arrays for compatibility with most ML frameworks.
    return np.array(dataX), np.array(dataY)


# get data
db = MongoDatabase()

# Y data
growth_data_1 = db.get_growth()
growth_data_2 = growth_data_1[['growth length   (cm)']]

# X data
environment_data_1 = db.get_environment(limit = 31200)
environment_data_2 = environment_data_1[['temp', 'humidity']]
environment_averaged = environment_data_2.groupby(environment_data_2.index // 100).mean(numeric_only=True).reset_index(drop=True)

# X+Y
training_data = pd.merge(environment_averaged, growth_data_2, left_index=True, right_index=True)

# split train, test
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(training_data)
X_data = data_normalized[:, :-1]
y_data = data_normalized[:, -1]
look_back = 24
X, Y = create_dataset(X_data, y_data, look_back)

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, shuffle=False)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, shuffle=False)

feature_size=len(X_train)

# RNN model
def create_rnn_model():
    model = keras.Sequential()
    model.add(SimpleRNN(64, input_shape=(look_back, 2), return_sequences=True))
    model.add(SimpleRNN(64))
    model.add(Dense(1))

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = 'mean_squared_error',
        metrics = [
            keras.metrics.MeanSquaredError(),
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.MeanAbsoluteError()
        ]
    )
    return model

def create_lstm_model():
    model = keras.Sequential()
    model.add(LSTM(units=64, input_shape=(look_back, 2)))
    model.add(Dense(1))

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = 'mean_squared_error',
        metrics = [
            keras.metrics.MeanSquaredError(),
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.MeanAbsoluteError()
        ]
    )
    return model

def create_transformer_model(look_back, input_features, num_heads=2, ff_dim=64, num_transformer_blocks=2, mlp_units=64, dropout_rate=0.1):
    inputs = Input(shape=(look_back, input_features))
    x = inputs

    # Stacked Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Multi-Head Self-Attention layer
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_features, dropout=dropout_rate)(x, x)
        attention_output = LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feed-Forward layer
        ff_output = Dense(ff_dim, activation='relu')(attention_output)
        ff_output = Dense(input_features)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

    # Final layers
    output = GlobalAveragePooling1D()(x)
    output = Dense(mlp_units, activation='relu')(output)
    output = Dropout(dropout_rate)(output)
    output = Dense(1, activation='linear')(output)

    model = Model(inputs=inputs, outputs=output)

    # Learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

#########################################

select_model = {
    'lstm' : {
        'name' : 'LSTM',
        'experiment': 'lstm_experiment', 
        'model_name' : 'lstm_model',
        'create' : 'create_lstm_model()',
        'fit' : {
            'x' : X_train, 
            'y' : Y_train, 
            'validation_data' : (X_val, Y_val),
            'epochs' : 128,
            'batch_size' : 32, 
            'verbose' : 2,
            'shuffle' : False
        }
    },
    'rnn' : {
        'name' : 'RNN',
        'experiment': 'rnn_experiment', 
        'model_name' : 'rnn_model',
        'create' : 'create_rnn_model()',
        'fit' : {
            'x' : X_train, 
            'y' : Y_train, 
            'validation_data' : (X_val, Y_val),
            'epochs' : 100
        }
    },
    'transformer': {
        'name': 'Transformer',
        'experiment': 'transformer_experiment',
        'model_name': 'transformer_model',
        'create': 'create_transformer_model(look_back, 2)',  
        'fit': {
            'x': X_train,
            'y': Y_train,
            'validation_data': (X_val, Y_val),
            'epochs': 100,  
            'batch_size': 64,  
            'verbose': 2,
            'shuffle': True  
        }
    }
}

# set_model
set_model = select_model['transformer']


# run MLFlow
mlflow.set_experiment(set_model['experiment'])

with mlflow.start_run():
    mlflow.tensorflow.autolog()

    # Define the model
    model = eval(set_model['create'])
    model.fit(**set_model['fit'])
    
    # Save the model
    print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.sklearn.log_model(model, set_model['model_name'])

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Visualize predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, predictions, color='blue', alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig("true_vs_predicted_values.png")
    plt.close()

    # Log scatter plot to mlflow
    mlflow.log_artifact("true_vs_predicted_values.png")

    # Plot a comparison between predicted and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test, label="Actual values", color='blue', alpha=0.5)
    plt.plot(predictions, label="Predicted values of "+set_model['name'], color='red', alpha=0.5)
    plt.title("Prediction vs Actual values")
    plt.savefig("comparison_plot.png")
    plt.close() 

    # Log comparison plot to mlflow
    mlflow.log_artifact("comparison_plot.png")

    # Make predictions on the train set
    train_predictions = model.predict(X_train)

    # Visualize predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_train, train_predictions, color='blue', alpha=0.5)
    plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig("true_vs_predicted_values_on_train_data.png")
    plt.close()

    # Log scatter plot to mlflow
    mlflow.log_artifact("true_vs_predicted_values_on_train_data.png")

    # Plot a comparison between predicted and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_train, label="Actual values", color='blue', alpha=0.5)
    plt.plot(train_predictions, label="Predicted values of "+set_model['name'], color='red', alpha=0.5)
    plt.title("Prediction vs Actual values")
    plt.savefig("comparison_plot_on_train_data.png")
    plt.close() 

    # Log comparison plot to mlflow
    mlflow.log_artifact("comparison_plot_on_train_data.png")

mlflow.end_run()