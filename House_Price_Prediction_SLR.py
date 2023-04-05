# -*- coding: utf-8 -*-

"""Import Libraries"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
"""Read Data"""
df=pd.read_csv('/content/USA_Housing.csv')
df.head()
df.info()
df.describe()
df.columns
"""Visualize Data"""
sns.pairplot(df)

x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df[['Price']]

"""Train Model"""

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=40)

from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train, y_train)

pred=lm.predict(X_test)

plt.scatter(y_test, pred)

"""Evaluate Results"""

from sklearn.metrics import r2_score
y_test=lm.predict(X_test)

print('R2-score: %2f' %r2_score(y_test, y_test))
print('Mean Absolute Error: %2f'%np.mean(np.absolute(y_test -y_test)))
