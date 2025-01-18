# Created by ivywang at 2025-01-18
import numpy as np
import pandas as pd
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

