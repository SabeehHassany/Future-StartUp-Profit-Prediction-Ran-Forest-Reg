"""
Random Forest Regression is essentially  multiple decision trees working together (hence random FOREST). It's one of the most effective regression models and usually outperforms all others aside from SVR. This dataset is a simple data from Kaggle that predicts the profit of 50 startups based on 4 predictor variables.

### Importing the libraries
These are the three go to libraries for most ML.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""### Importing the dataset
Simple dataset import using Pandas dataframe and iloc to assign our independent variable(s) (everything besides the last column) and our dependent variable (the last column). The name of the dataset has to be updated and it must be in the same folder as your .py file or uploaded on Jupyter Notebooks or Google Collab.
"""

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""### Encoding categorical data
Index 3 had categorical data that had to be converted using OneHotEncoding.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

"""### Splitting the dataset into the Training set and Test set
Because of the smaller dataset and less variables for testing, I used a 80/20 split. The random state is tuned to 0 for consistency sakes.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""### Training the Random Forest Regression model on the whole dataset
Here, 'n_estimators' is the value assigned to how many trees we have. 10 is usually the go to but with model tweaking and grid search and k fold, you can find the optimal amount for your specific problem.
"""

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

"""### Predicting the Test set results
By using the concatenate function I display the predicted values and  actual values in a side by side 2D array through '(len(y_wtv), 1))' for easy viewing.
"""

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=0, suppress=True)
print(np.concatenate((y_pred.astype(int).reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""### Evaluating Model Performance
We use two metrics to evaluate our model performance, r^2 being the more superior. These are both simple to understand and are covered in one of my Medium articles! This model acheives a r2 of .96 (.01 better than decision tree regression) with only 50 instances of testing data! Pretty impressive.
"""

from sklearn.metrics import r2_score, mean_squared_error as mse
print("r^2: " + str(r2_score(y_test, y_pred)))
print("MSE: " + str(mse(y_test, y_pred)))