import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from yahoo_fin import stock_info as si
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import uuid
import os
from datetime import date
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing as sk
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from IPython.display import clear_output

st.write("""
# Stock prediction web application
This app predicts the future stock price of a selected ticker based on historical data. 
""")

st.sidebar.header('User Input Parameters')
currentYear = datetime.now().year


class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()
        animate(epoch)


with st.sidebar.form(key='Form1'):
    stock_ticker = st.selectbox('Stock ticker (S&P 500)', (si.tickers_sp500()))
    future_steps = st.slider('Future steps', 1, 10, 5)
    year_from = st.slider('Year from', 1980, currentYear - 1, 1990)
    data = {'future_steps': future_steps,
            'stock_ticker': stock_ticker,
            'year_from': year_from}
    features = pd.DataFrame(data, index=[0])
    submit_button = st.form_submit_button(label='Save')

df_ui = features

st.subheader('User Input parameters')
st.write(df_ui)

st.subheader('Stock ticker information')
info = si.get_quote_table(df_ui['stock_ticker'][0])
st.write(info)

st.subheader('Dataframe sample : ' + df_ui['stock_ticker'][0])
df = si.get_data(df_ui['stock_ticker'][0])
df = df[~df.index.duplicated(keep='first')]
df["date"] = df.index
df = df[str(df_ui['year_from'][0]) + '0101':str(currentYear) + '1231']
df['future'] = df['adjclose'].shift(-df_ui['future_steps'][0])
df = df.dropna()
st.write(df.tail())

st.write("The price per share of " + df_ui['stock_ticker'][0] + " illustrated from the dataframe")
fig, ax = plt.subplots(figsize=(24, 12))
ax.plot(df['adjclose'], c='b')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(["Adjusted Close"], loc="upper left")
st.pyplot(fig)

st.subheader('LSTM Training')


with st.form(key='Config'):
    epochs = st.slider('Epochs', 1, 1000, 882, help="Every epoch means a complete pass through the training dataset. "
                                                    "An early stopping callback is used, which stops the model at the"
                                                    " lowest error possible.")
    batch_size = st.slider('Batch Size', 16, 320, 64, step=16, help="The batch size is the number of samples that are "
                                                                    "passed through the model at one time.")
    units = st.slider('Units', 1, 300, 130, help="These units represent how many neurons are contained in the model’s "
                                                 "layers.")
    window_length = st.slider('Window Length', 1, 300, 50, help="The sequential length of the input data")

    data = {'epochs': epochs,
            'batch_size': batch_size,
            'units': units,
            'window_length': window_length}
    config = pd.DataFrame(data, index=[0])
    submit_button = st.form_submit_button(label='Update Settings')
    if submit_button:
        st.success("Settings changed! You can start training the model")


if st.button("Start training LSTM"):
    features = ['adjclose', 'volume', 'open', 'high', 'low']
    target = ['future']

    # DATA TRANSFORMATION
    transformer_features = sk.MinMaxScaler().fit(df[features])
    transformer_target = sk.MinMaxScaler().fit(df[target])
    df_copy = df.copy()
    df_copy[features] = transformer_features.fit_transform(df_copy[features])
    df_copy[target] = transformer_target.fit_transform(df_copy[target])

    # DATA SPLITTING
    X = df_copy[features]
    y = df_copy[target]
    T = config['window_length'][0]
    # 80% train 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    prepend_features = X_train.iloc[-(T - 1):]
    X_test = pd.concat([prepend_features, X_test], axis=0)

    # Create sequences of T timesteps
    X_train_lstm, y_train_lstm = [], []
    for i in range(y_train.shape[0] - (T - 1)):
        X_train_lstm.append(X_train.iloc[i:i + T].values)
        y_train_lstm.append(y_train.iloc[i + (T - 1)])
    X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm).reshape(-1, 1)
    print(f'Train data dimensions: {X_train_lstm.shape}, {y_train_lstm.shape}')

    X_test_lstm, y_test_lstm = [], []
    for i in range(y_test.shape[0]):
        X_test_lstm.append(X_test.iloc[i:i + T].values)
        y_test_lstm.append(y_test.iloc[i])
    X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm).reshape(-1, 1)

    # MODEL CREATION
    model = Sequential()
    cell = LSTM
    units = config['units'][0]
    bidirectional = False
    n_features = len(features)
    optimizer = "adam"
    loss = "huber_loss"
    dropout = 0.11661839507929572
    sequence_length = T
    BATCH_SIZE = config['batch_size'][0]
    EPOCHS = config['epochs'][0]

    # first layer
    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
    model.add(Dropout(dropout))

    # last layer
    model.add(cell(units, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation='linear'))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    # MODEL TRAINING
    model_name = df_ui['stock_ticker'][0] + "_LSTM_" + str(date.today()) + '_' + str(uuid.uuid4())
    # checkpointer = ModelCheckpoint(model_name + ".h5", save_weights_only=True,
    #                                save_best_only=True,
    #                                verbose=1)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)

    st.write("The model's performance is being updated for every epoch")
    fig, ax = plt.subplots()
    max_x = 1
    max_rand = 1
    x = np.arange(0, max_x)
    ax.set_ylim(0, max_rand)
    line, = ax.plot(x, np.random.randint(0, max_rand, max_x))
    the_plot = st.pyplot(plt)

    def init():  # give a clean slate to start
        line.set_ydata([np.nan] * len(x))

    def animate(i):  # update the y values (every 1000ms)
        line.set_ydata(np.random.randint(0, max_rand, max_x))
        the_plot.pyplot(plt)

    init()

    with st.spinner('Training takes a while please wait...'):
        history = model.fit(X_train_lstm, y_train_lstm,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(X_test_lstm, y_test_lstm),
                            callbacks=[earlyStop, PlotLearning()],
                            verbose=1)
    st.success("Model Trained.")
    st.write("Displaying results...")
    y_pred = model.predict(X_test_lstm)

    y_pred_inv = transformer_target.inverse_transform(y_pred)
    y_test[target] = transformer_target.inverse_transform(y_test[target])
    X_test[features] = transformer_features.inverse_transform(X_test[features])
    y_test['pred'] = y_pred_inv

    y_test_inv = transformer_target.inverse_transform(y_test_lstm)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)*100
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    mda = np.mean((np.sign(y_test_inv[1:] - y_test_inv[:-1]) == np.sign(y_pred_inv[1:] - y_test_inv[:-1])).astype(int))

    fut_data = {
        'RMSE': rmse,
        'MAPE': mape,
        'MAE': mae,
        'MDA': mda,
        'R2': r2}
    df_fut = pd.DataFrame(fut_data, index=[df_ui['stock_ticker'][0]])
    st.write(df_fut)

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.plot(df['future'][:y_test.index[0]], c='b')
    ax.plot(y_test['future'], c='navy')
    ax.plot(y_test['pred'], c='darkred')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(["Train Data", "Test Data", "Predicted Price"])
    st.pyplot(fig)

    merged_df = pd.merge(y_test, X_test, left_index=True, right_index=True)
    merged_df['movement_pred'] = np.where(merged_df.pred >= merged_df.adjclose, 1, 0)
    merged_df['pred_percentage'] = ((merged_df.pred / merged_df.adjclose) - 1) * 100
    merged_df['movement'] = np.where(merged_df.future >= merged_df.adjclose, 1, 0)
    merged_df['movement_percentage'] = ((merged_df.future / merged_df.adjclose) - 1) * 100
    merged_df['date'] = merged_df.index

    investment_start = 1000
    current = 0
    buy_dates = {}
    sell_dates = {}
    merged_df['pred_percentage'] = ((merged_df.pred / merged_df.adjclose) - 1) * 100
    for i, r in merged_df.iterrows():
        if r['pred_percentage'] > 2:
            current += investment_start * 0.75
            investment_start -= investment_start * 0.75
            res = current * (r['movement_percentage'] / 100)
            buy_dates[r['date']] = r['future']
            current += res
        elif 2 > r['pred_percentage'] >= 1:
            current += investment_start * 0.5
            investment_start -= investment_start * 0.5
            res = current * (r['movement_percentage'] / 100)
            buy_dates[r['date']] = r['future']
            current += res
        elif 1 > r['pred_percentage'] >= 0.5:
            current += investment_start * 0.25
            investment_start -= investment_start * 0.25
            res = current * (r['movement_percentage'] / 100)
            buy_dates[r['date']] = r['future']
            current += res
        elif 0.5 > r['pred_percentage'] >= 0.05:
            current += investment_start * 0.05
            investment_start -= investment_start * 0.05
            res = current * (r['movement_percentage'] / 100)
            buy_dates[r['date']] = r['future']
            current += res
        else:
            if current != 0:
                investment_start += current
                sell_dates[r['date']] = r['future']
                current = 0

    investment_data = {
        'Investment start': '1000',
        'Investment end': investment_start + current,
        'Profit/Loss': (investment_start + current) - 1000,
        'Total Indicators': len(buy_dates) + len(sell_dates),
        'Buy Indicators': len(buy_dates),
        'Sell Indicators': len(sell_dates)}
    df_investment = pd.DataFrame(investment_data, index=[df_ui['stock_ticker'][0]])

    st.write("# Investment experiment on test data")
    st.latex(r'''
            \text { investment }=\left\{\begin{array}{l}
            75 \% \text { if prediction } \% \text { change } \geq 2 \% \\
            50 \% \text { if } 2 \%>\text { prediction } \% \text { change } \geq 1 \% \\
            25 \% \text { if } 1 \%>\text { prediction } \% \text { change } \geq 0.5 \% \\
            5 \% \text { if } 0.05 \%>\text { prediction } \% \text { change } \geq 0.05 \%
            \end{array}\right.
         ''')
    st.write("The investment amount depends on the confidence of the prediction percentage change of the model. If "
             "none of the above criteria is met, a sell trigger is initiated, and the result is added to the "
             "investment variable. This process is repeated for all of the test data. A graph is outputted containing "
             "buy and sell indicators across the test data.")
    st.write(df_investment)
