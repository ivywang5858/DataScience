# Created by ivywang at 2025-01-17
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv('real_estate_price_size.csv')
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