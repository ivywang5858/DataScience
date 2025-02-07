# Created by ivywang at 2025-01-30
import simfin as sf
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore')

current_path = os.path.abspath(os.path.dirname(__file__))

def get_fi_data():
    sf.set_api_key('8098dae9-e55b-4c33-8b74-2df2a9d104be')
    sf.set_data_dir('~PycharmProjects/Udemy/DataScience/simfin_data/')
    income_df = sf.load_income(variant= "quarterly",market="us").reset_index()
    print(income_df)
    return
# get_fi_data()

# Data Scaling using MinMaxScaler()
def ML_DL():
    FILE_NAME = "NeuralNetwork_financial_data.csv"
    df = pd.read_csv(f"{current_path+'/Data/'}{FILE_NAME}")
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

def XGBoost():
    import xgboost as xgb

    FILE_NAME = "NeuralNetwork_financial_data.csv"
    df = pd.read_csv(f"{current_path + '/Data/'}{FILE_NAME}")
    apple_df = df[df['Ticker'] == 'AAPL']

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

    # Train an XGBoost regressor model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                 learning_rate=0.1,
                                 max_depth=3,
                                 n_estimators=200)
    xgb_model.fit(X_train, y_train)
    # Make predictions on the test data
    y_predict = xgb_model.predict(X_test)

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
    # plt.show()

    # GridSearchCV performs exhaustive search over a specified list of parameters
    # You provide the algorithm with the hyperparameters and values you would like to experiment with
    # Note that you will have the following number of combinations: 4 * 4 * 4 * 3 = 192
    # We will run each combination 5 times since we set the cross validation = 5
    # Total number of runs = 192 * 5 = 960 fits

    from sklearn.model_selection import GridSearchCV
    # Specify the parameters grid
    parameters_grid = {'max_depth': [1, 3, 5, 10],
                       'learning_rate': [0.01, 0.1, 0.5, 1],
                       'n_estimators': [10, 50, 100, 200],
                       'subsample': [0.5, 0.75, 1.0]}

    # Note that "neg_mean_squared_error" is used for scoring since our goal is to minimize the error
    # GridSearchCV() ranks all algorithms (estimators) and specifies which one is the best
    # cv stands for the number of cross-validation folds which is set to 5 by default
    # Verbose controls the verbosity: the higher the number, the more messages to be displayed
    xgb_gridsearch = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),
                                  param_grid=parameters_grid,
                                  scoring='neg_mean_squared_error',
                                  cv=5,
                                  verbose=5)
    # Let's fit the model
    xgb_gridsearch.fit(X_train, y_train)
    # Indicate best parameters after grid search optimization
    print(xgb_gridsearch.best_params_)
    print(xgb_gridsearch.best_estimator_)
    # Generate predictions based on the optimal model parameters
    y_predict = xgb_gridsearch.predict(X_test)

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
    return

def Basic_NN():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    observations = 1000
    xs = np.random.uniform(-10,10,(observations,1))
    zs = np.random.uniform(-10,10,(observations,1))
    # form 2 cols
    inputs = np.column_stack((xs,zs))

    # create the target function
    noise = np.random.uniform(-1,1,(observations,1))
    targets = 2*xs - 3*zs + 5 + noise # y = xw +b

    # In order to use the 3D plot, the objects should have a certain shape, so we reshape the targets.
    # The proper method to use is reshape and takes as arguments the dimensions in which we want to fit the object.
    targets = targets.reshape(observations, )

    # Plotting according to the conventional matplotlib.pyplot syntax

    # Declare the figure
    fig = plt.figure()

    # A method allowing us to create the 3D plot
    ax = fig.add_subplot(111, projection='3d')

    # Choose the axes.
    ax.plot(xs, zs, targets)

    # Set labels
    ax.set_xlabel('xs')
    ax.set_ylabel('zs')
    ax.set_zlabel('Targets')
    ax.view_init(azim=100)
    plt.show()

    # We reshape the targets back to the shape that they were in before plotting.
    # This reshaping is a side-effect of the 3D plot. Sorry for that.
    targets = targets.reshape(observations, 1) # y = xw +b

    init_range = 0.1
    weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))
    biases = np.random.uniform(low=-init_range, high=init_range, size=1)
    learning_rate = 0.02

    # Train the model
    for i in range(500):
        outputs = np.dot(inputs, weights) + biases
        deltas = outputs - targets
        # We are considering the L2-norm loss, but divided by 2, so it is consistent with the lectures.
        # Moreover, we further divide it by the number of observations.
        # This is simple rescaling by a constant. We explained that this doesn't change the optimization logic,
        # as any function holding the basic property of being lower for better results, and higher for worse results
        # can be a loss function.
        loss = np.sum(deltas ** 2) / 2 / observations
        deltas_scaled = deltas / observations
        weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
        biases = biases - learning_rate * np.sum(deltas_scaled)

    plt.plot(outputs, targets)
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.show()

    return


# ML_DL()
# XGBoost()
Basic_NN()


