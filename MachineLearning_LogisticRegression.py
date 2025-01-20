# Created by ivywang at 2025-01-18
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Logistic Regression Assumptions
# 1. Non Linearity (deleted)
# 2. No endogeneity
# 3. Normality and homoscedasticity
# 4. No autocorrelation
# 5. No multicollinearity

# Logistic Model --> P(X) = e*(b0+b1x1+...+bkxk)/(1+e*(b0+b1x1+...+bkxk))
# Odds = P(occurring)/P(not occurring)
# Logit Model --> Log(odds) = b0+b1x1+...+bkxk

# MLE: Maximum likelihood estimation -- a function which estimates how likely it is
# that the model at hand describes the real underlying relationship of the variables
# MLE tries to maximize the log likelihood. The computer is going through different
# values, until it finds a model, for which the log likelihood is the highest.
# When it can no longer improve it, it will just stop the optimization
# Log-likelihood: the bigger, the better

# LL-Null (log likelihood-null):
# the log-likelihood of a model which has no independent variables
# y = b0*1

# LLR (log likelihood ratio):
# measures if our model is statistically different from LL-null, a.k.a. a useless model
# check if model is significantly different from LL-Null

# Pseudo R-squared (McFadden's R-squared):
# a good Pseudo R-squared is somewhere between 0.2 and 0.4.
# This measure is mostly useful for comparing variations of the same model

# delta(odds) = e*(bk)
# delta(odds) = odds2/odds1

# The data is based on the marketing campaign efforts of a Portuguese banking institution.
# The classification goal is to predict if the client will subscribe a term deposit (variable y).
def Logistic_Regression():
    raw_data = pd.read_csv('Logit_Example_bank_data.csv')
    data = raw_data.copy()
    data = data.drop(['Unnamed: 0'], axis=1)
    data['y'] = data['y'].map({'yes':1,'no':0})
    y = data['y']
    x1 = data['duration']
    # Simple Logistic Regression
    x = sm.add_constant(x1)
    reg_log = sm.Logit(y,x)
    results_log = reg_log.fit()
    # Get the regression summary
    print(results_log.summary())

    # Create a scatter plot of x1 (Duration, no constant) and y (Subscribed)
    plt.scatter(x1, y, color='C0')
    # Don't forget to label your axes!
    plt.xlabel('Duration', fontsize=20)
    plt.ylabel('Subscription', fontsize=20)
    plt.show()
    return

def Binary_Predictors_in_Logistic_Regression():
    raw_data = pd.read_csv('Logit_Bank_data.csv')
    # We make sure to create a copy of the data before we start altering it. Note that we don't change the original data we loaded.
    data = raw_data.copy()
    # Removes the index column thata comes with the data
    data = data.drop(['Unnamed: 0'], axis=1)
    # We use the map function to change any 'yes' values to 1 and 'no'values to 0.
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    y = data['y']
    x1 = data['duration']
    x = sm.add_constant(x1)
    reg_log = sm.Logit(y, x)
    results_log = reg_log.fit()
    # Get the regression summary
    print(results_log.summary())

    # Create a scatter plot of x1 (Duration, no constant) and y (Subscribed)
    plt.scatter(x1, y, color='C0')
    # Don't forget to label your axes!
    plt.xlabel('Duration', fontsize=20)
    plt.ylabel('Subscription', fontsize=20)
    plt.show()
    # the odds of duration are the exponential of the log odds from the summary table
    np.exp(0.0051)
    # The odds of duration are pretty close to 1.
    # This tells us that although duration is a significant predictor,
    # a change in 1 day would barely affect the regression.

    np.set_printoptions(formatter={'float':lambda x: "{0:0.2f}".format(x)})
    print(results_log.predict())
    cm_df = pd.DataFrame(results_log.pred_table())
    cm_df.columns = ['Predicted 0','Predicted 1']
    cm_df = cm_df.rename(index={0:'Actual 0',1:'Actual 1'})
    print(cm_df)
    return

# sm.LogitResults.pred_table()
# returns a table which compares predicted and actual values
# Confusion Matrix
# accuracy = (TP+TN)/(sum of all)

def confusion_matrix(data, actual_values, model):
    # Confusion matrix
    # Parameters
    # ----------
    # data: data frame or array
    # data is a data frame formatted in the same way as your input data (without the actual values)
    # e.g. const, var1, var2, etc. Order is very important!
    # actual_values: data frame or array
    # These are the actual values from the test_data
    # In the case of a logistic regression, it should be a single column with 0s and 1s

    # model: a LogitResults object
    # this is the variable where you have the fitted model
    # e.g. results_log in this course
    # ----------

    # Predict the values using the Logit model
    pred_values = model.predict(data)
    # Specify the bins
    bins = np.array([0, 0.5, 1])
    # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
    # if they are between 0.5 and 1, they will be considered 1
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    # Calculate the accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    # Return the confusion matrix and
    return cm, accuracy


# Overfitting and how to resolve it
# Overfitting: our training has focused on the particular training set so much,
# it has "missed the point"
# Underfitting: the model has not captured the underlying logic of the data
# One popular solution to overfitting is to split the initial dataset into two: training and test
# test the model on the testing data by creating a confusion matrix and assessing the accuracy

# Test the model
# 1. Use our model to make predictions based on the test data
# 2. Compare those with the actual outcome
# 3. Calculate the accuracy
# 4. Create a confusion matrix
def Test_Model():
    raw_data = pd.read_csv('Logit_Bank_data.csv')
    # We make sure to create a copy of the data before we start altering it. Note that we don't change the original data we loaded.
    data = raw_data.copy()
    # Removes the index column that comes with the data
    data = data.drop(['Unnamed: 0'], axis=1)
    # We use the map function to change any 'yes' values to 1 and 'no'values to 0.
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    y = data['y']
    x1 = data['duration']
    x = sm.add_constant(x1)
    reg_log = sm.Logit(y, x)
    results_log = reg_log.fit()
    # Get the regression summary
    print(results_log.summary())

    # To avoid writing them out every time, we save the names of the estimators of our model in a list.
    estimators = ['interest_rate', 'credit', 'march', 'previous', 'duration']

    X1_all = data[estimators]
    y = data['y']
    X_all = sm.add_constant(X1_all)
    reg_logit = sm.Logit(y, X_all)
    results_logit = reg_logit.fit()
    print(results_logit.summary2())
    print(confusion_matrix(X_all, y, results_logit))

    # Test the model with the new data
    # We have to load data our model has never seen before.
    raw_data2 = pd.read_csv('Logit_Bank_data_testing.csv')
    data_test = raw_data2.copy()
    # Removes the index column thata comes with the data
    data_test = data_test.drop(['Unnamed: 0'], axis=1)
    # Coverting the outcome variable into 1s and 0s again.
    data_test['y'] = data_test['y'].map({'yes': 1, 'no': 0})
    y_test = data_test['y']
    # We already declared a list called 'estimators' that holds all relevant estimators for our model.
    X1_test = data_test[estimators]
    X_test = sm.add_constant(X1_test)
    # Determine the Confusion Matrix and the accuracy of the model with the new data. Note that the model itself stays the same (results_logit).
    # test accuracy
    print(confusion_matrix(X_test, y_test, results_logit))
    # Looking at the test acccuracy we see a number which is a tiny but lower: 86.04%, compared to 86.29% for train accuracy.
    # In general, we always expect the test accuracy to be lower than the train one. If the test accuracy is higher, this is just due to luck.
    return

# Logistic_Regression()
# Binary_Predictors_in_Logistic_Regression()
Test_Model()

