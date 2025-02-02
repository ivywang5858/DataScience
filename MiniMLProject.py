# Created by ivywang at 2025-01-30
import simfin as sf
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sf.set_api_key('8098dae9-e55b-4c33-8b74-2df2a9d104be')
sf.set_data_dir('~PycharmProjects/Udemy/DataScience/simfin_data/')

def get_fi_data():
    income_df = sf.load_income(variant= "quarterly",market="us").reset_index()
    print(income_df)
    return
# get_fi_data()

# Data Scaling using MinMaxScaler()
def ML_DL():
    df = pd.read_csv('MiniML_financial_data.csv')
    apple_df = df[df['Ticker']=='AAPL']

    # Python has a module dedicated to deal with date and time data known as datetime
    apple_df['Publish Date'] = pd.to_datetime(apple_df['Publish Date'])
    # Sorting the DataFrame in an ascending order based on the "Publish Date" column
    apple_df.sort_values(by='Publish Date', ascending=True, inplace=True)
    # Let's drop the following columns from the Pandas DataFrame
    cols_to_drop = ['Ticker', 'Sector', 'Industry', 'Company Name', 'Report Date', 'Currency',
                    'Fiscal Year', 'Publish Date', 'Restated Date']
    apple_df = apple_df.drop(columns=cols_to_drop)
    # Let's display the one-hot encoded version of the "Fiscal Period" column
    fiscal_encoded = pd.get_dummies(apple_df['Fiscal Period'])
    # Drop the 'Fiscal Period' column from the Pandas DataFrame
    apple_df = apple_df.drop('Fiscal Period', axis=1)
    # Concatenate the original DataFrame and the one-hot encoded
    apple_df = pd.concat([apple_df, fiscal_encoded], axis=1)

    # Split the data into inputs "X" and outputs "y"
    X = apple_df.drop('% Change in Quarterly EPS (Target Output)', axis=1)
    y = apple_df['% Change in Quarterly EPS (Target Output)']

    # Perform data train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # MinMaxScaler scales the data to a range between 0 and 1
    # Apply data scaling to the training and testing datasets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform output data scaling
    scaler = MinMaxScaler()
    scaler.fit(pd.DataFrame(y_train))
    # Apply data scaling to the training and testing datasets
    y_train_scaled = scaler.transform(pd.DataFrame(y_train))
    y_test_scaled = scaler.transform(pd.DataFrame(y_test))

    # Let's build our ANN using Keras API
    # Keras sequential model allows for building ANNs which consist of sequential layers (input, hidden..etc.)
    # Dense means each neuron in the layer is fully connected to all the neurons in the previous layer
    # Normalization layer normalizes its inputs by applies a transformation to maintain the mean output close to 0
    # and standard deviation close to 1.

    from keras.models import Sequential
    from keras.layers import Dense, Normalization, Dropout
    ANN_model = Sequential()
    ANN_model.add(Normalization(input_shape=[X_train.shape[1], ], axis=None))
    ANN_model.add(Dense(1024, activation='relu'))
    ANN_model.add(Dropout(0.3))
    ANN_model.add(Dense(512, activation='relu'))
    ANN_model.add(Dropout(0.3))
    ANN_model.add(Dense(256, activation='sigmoid'))
    ANN_model.add(Dropout(0.3))
    ANN_model.add(Dense(32, activation='sigmoid'))
    ANN_model.add(Dropout(0.3))
    ANN_model.add(Dense(units=1, activation='linear'))

    # Let's obtain the model summary
    ANN_model.summary()
    # Adam optimizer is the extended version of stochastic gradient descent algorithm
    # It works well in deep learning applications and ANNs training
    ANN_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mean_squared_error')

    # Let's fit the model using 500 epochs
    history = ANN_model.fit(X_train_scaled, y_train_scaled, epochs=500)

    # Let's generate model predictions using the testing dataset and then scale the data back to its original range values
    y_predict_scaled = ANN_model.predict(X_test_scaled)
    y_predict = scaler.inverse_transform(y_predict_scaled)
    y_test = scaler.inverse_transform(y_test_scaled)

    # Let's generate various regression metrics by comparing "y_predict" vs. actual outputs "y_test" (ground truth data)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    RMSE = float(np.sqrt(mean_squared_error(y_test, y_predict)))
    MSE = mean_squared_error(y_test, y_predict)
    MAE = mean_absolute_error(y_test, y_predict)

    print('Root Mean Squared Error (RMSE) =', RMSE, '\nMean Squared Error (MSE) =', MSE,
          '\nMean Absolute Error (MAE) =', MAE)

    # Plot model predictions "y_predict" vs. actual outputs "y_test" (ground truth data)
    plt.figure(figsize=(13, 8))
    plt.plot(y_predict, y_test, 'o', color='r', markersize=10)
    plt.xlabel('Model Predictions')
    plt.ylabel('Actual Values (Ground Truth)')
    plt.title('Model Predictions Vs. Actual Values (Ground Truth)')
    plt.show()
    return

ML_DL()


