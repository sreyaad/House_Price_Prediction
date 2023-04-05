# House_Price_Prediction
## Problem Statement-
A real state agents want help to predict the house price for regions in the USA. He gave you the dataset to work on and you decided to use the Linear Regression Model. Create a model that will help him to estimate of what the house would sell for.

## Importing Libraries
For solving the problem, we will first import the necessary libraries- *numpy, pandas, seaborn and matplotlib*

## Read Data-
For reading the csv file, we will use the read_csv function of pandas, and to display the information of the data, we use **.info** functon.  

## Data Visualization-
For Data Visualization, we use **pairplot** function of seaborn.

## Split training and testing data-
- We used sklearn **train_test_split()** to divide our dataset into training data and test data. 
- We used 80% of the data for training and the rest 20% for testing.

## Model Training
We imported **LinearRegression** from sklearn.linear_model, and then fit the training data into it.

## Results-
After testing the model on our test data, we obtained **100%** accuracy. We had used r2_score and Mean-Squared Error as our Evaluation metrics.
