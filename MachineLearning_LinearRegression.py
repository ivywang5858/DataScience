# Created by ivywang at 2025-01-17
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.linear_model import LinearRegression

# Linear Regression Assumptions
# 1. Linearity
# 2. No endogeneity (xi & residual error no correlation)
# 3. Normality and homoscedasticity
# 4. No autocorrelation
# 5. No multicollinearity

def simple_Linear_Regression():
    ##### Sinple Linear Regression ###################
    data = pd.read_csv('LR_real_estate_price_size_year.csv')
    # print(data.describe())
    x = data['size']
    y = data['price']
    plt.scatter(x,y)
    plt.xlabel('Size',fontsize=20)
    plt.ylabel('Price',fontsize=20)
    # plt.show()

    # Transform the inputs into a matrix (2D object)
    x_matrix = x.values.reshape(-1,1)
    reg = LinearRegression()
    reg.fit(x_matrix,y)

    # Calc R-sqaured
    reg.score(x_matrix,y)
    print("The intercept is: {}".format(reg.intercept_))
    print("The coefficient is: {}".format(reg.coef_))

    # Make the prediction
    pred = reg.predict(np.array(750).reshape(-1,1))
    print("The prediction is: {}".format(pred))
    return

def multi_Linear_Regression():
    ##### Multi Linear Regression ###################
    data = pd.read_csv('LR_real_estate_price_size_year.csv')
    print(data.describe())
    x = data[['size', 'year']]
    y = data['price']
    reg = LinearRegression()
    reg.fit(x, y)
    print("The intercept is: {}".format(reg.intercept_))
    print("The coefficient is: {}".format(reg.coef_))

    # Calc R-sqaured
    r2 = reg.score(x, y)
    print("The R-sqauredt is: {}".format(r2))
    # Calc Adjusted R-sqaured
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print("The adjusted R-sqauredt is: {}".format(adjusted_r2))

    # Make the prediction
    # Find the predicted price of an apartment that has a size of 750 sq.ft. from 2009
    pred = reg.predict([[750, 2009]])
    print(pred)

    # F-Regression (feature selection)
    from sklearn.feature_selection import f_regression
    f_regression(x, y)
    f_statistics = f_regression(x, y)[0]
    p_values = f_regression(x, y)[1]
    p_values.round(3)
    reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
    reg_summary['Coefficients'] = reg.coef_
    reg_summary['p-values'] = p_values.round(3)
    print(reg_summary)
    # It seems that 'Year' is not event significant, therefore we should remove it from the model.
    return

# Standardize the data --> (x-mean)/std
def feature_scaling():
    data = pd.read_csv('LR_real_estate_price_size_year.csv')
    print(data.describe())
    x = data[['size', 'year']]
    y = data['price']

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    reg = LinearRegression()
    reg.fit(x_scaled, y)
    print("The intercept is: {}".format(reg.intercept_))
    print("The coefficient is: {}".format(reg.coef_))
    # Calc R-sqaured
    r2 = reg.score(x_scaled, y)
    print("The R-sqauredt is: {}".format(r2))
    # Calc Adjusted R-sqaured
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print("The adjusted R-sqauredt is: {}".format(adjusted_r2))
    ## It seems the the R-squared is only slightly larger than the Adjusted R-squared,
    # implying that we were not penalized a lot for the inclusion of 2 independent variables.

    # Make the prediction using standardization
    # Find the predicted price of an apartment that has a size of 750 sq.ft. from 2009
    new_data = [[750, 2009]]
    new_data_scaled = scaler.transform(new_data)
    pred = reg.predict(new_data_scaled)
    print(pred)

    # F-Regression (feature selection)
    from sklearn.feature_selection import f_regression
    f_regression(x_scaled, y)
    p_values = f_regression(x, y)[1]
    p_values.round(3)
    reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
    reg_summary['Coefficients'] = reg.coef_
    reg_summary['p-values'] = p_values.round(3)
    print(reg_summary)
    # It seems that 'Year' is not event significant, therefore we should remove it from the model
    return

# Dummy variables and Variance Inflation Factor
def dummy_VIF():
    raw_data = pd.read_csv('LR_Dummy_and_VIF.csv')
    print(raw_data.describe(include='all'))
    data = raw_data.drop(['Model'], axis=1)
    no_value = data.isnull().sum()
    print(no_value)
    # remove the missing value
    data_no_mv = data.dropna(axis=0)
    print(data_no_mv.describe(include='all'))

    # show the distribution and deal with the outliers
    fig, axs = plt.subplots(4,2,figsize=(18, 15))
    # Price
    sns.histplot(data_no_mv['Price'],ax=axs[0, 0],kde=True,stat="density")
    axs[0,0].set_title('Price with the outlier')
    q = data_no_mv['Price'].quantile(0.99)
    data_1 = data_no_mv[data_no_mv['Price'] < q]
    data_1.describe(include='all')
    sns.histplot(data_1['Price'],ax=axs[0, 1],kde=True,stat="density")
    axs[0,1].set_title('Price without the outlier')

    # Mileage
    sns.histplot(data_no_mv['Mileage'],ax=axs[1, 0],kde=True,stat="density")
    axs[1,0].set_title('Mileage with the outlier')
    q = data_1['Mileage'].quantile(0.99)
    data_2 = data_1[data_1['Mileage'] < q]
    sns.histplot(data_2['Mileage'],ax=axs[1, 1],kde=True,stat="density")
    axs[1,1].set_title('Mileage without the outlier')

    # EngineV
    sns.histplot(data_no_mv['EngineV'],ax=axs[2, 0],kde=True,stat="density")
    axs[2,0].set_title('EngineV with the outlier')
    data_3 = data_2[data_2['EngineV'] < 6.5]
    sns.histplot(data_3['EngineV'],ax=axs[2, 1],kde=True,stat="density")
    axs[2,1].set_title('EngineV without the outlier')

    # Year
    sns.histplot(data_no_mv['Year'],ax=axs[3, 0],kde=True,stat="density")
    axs[3,0].set_title('Year with the outlier')
    q = data_3['Year'].quantile(0.01)
    data_4 = data_3[data_3['Year'] > q]
    sns.histplot(data_4['Year'],ax=axs[3, 1],kde=True,stat="density")
    axs[3,1].set_title('Year without the outlier')
    # plt.show()
    data_cleaned = data_4.reset_index(drop=True)
    data_cleaned.describe(include='all')

    ## Checking the OLS assumptions - heteroskedasticity
    fig2, ax2 = plt.subplots(2, 3,  figsize=(15, 6))
    ax2[0,0].scatter(data_cleaned['Year'], data_cleaned['Price'])
    ax2[0,0].set_title('Price and Year')
    ax2[0,1].scatter(data_cleaned['EngineV'], data_cleaned['Price'])
    ax2[0,1].set_title('Price and EngineV')
    ax2[0,2].scatter(data_cleaned['Mileage'], data_cleaned['Price'])
    ax2[0,2].set_title('Price and Mileage')

    log_price = np.log(data_cleaned['Price'])
    # take log of price
    data_cleaned['log_price'] = log_price
    ax2[1,0].scatter(data_cleaned['Year'], data_cleaned['log_price'])
    ax2[1,0].set_title('Log Price and Year')
    ax2[1,1].scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
    ax2[1,1].set_title('Log Price and EngineV')
    ax2[1,2].scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
    ax2[1,2].set_title('Log Price and Mileage')
    # plt.show()

    # Multicollinearity - check with VIF
    data_cleaned.columns.values
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif["features"] = variables.columns
    print(vif)
    # drop the col when VIF > 10
    data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)

    # Create dummy variables
    data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
    print(data_with_dummies.head())
    data_with_dummies.columns.values
    cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
            'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
            'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
            'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
            'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
    data_preprocessed = data_with_dummies[cols]
    print(data_preprocessed.head())
    return



# simple_Linear_Regression()
# multi_Linear_Regression()
# feature_scaling()
dummy_VIF()