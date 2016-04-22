import pandas as pd
import numpy as np
from sklearn import cross_validation as cv

# Calculate intercept & slope using a closed form equation
def simple_linear_regression(input_feature, output):
    N = len(input_feature)
    x = np.array(input_feature)
    y = np.array(output)

    sum_x = np.sum(x)
    sum_xx = np.sum(np.multiply(x, x))
    sum_y = np.sum(y)
    sum_xy = np.sum(np.multiply(x, y))

    slope = (sum_xy - sum_x*sum_y/N)/(sum_xx - sum_x*sum_x/N)
    intercept = sum_y/N - slope*sum_x/N

    return (intercept, slope)

# Return the predicted output
def get_regression_predictions(input_feature, intercept, slope):
    return intercept + slope*input_feature

# Calculate the residual sum of squares
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    x = np.array(input_feature)
    y_hat = intercept + slope*x
    y = np.array(output)
    diff =np.subtract(y_hat, y)
    rss = np.sum(np.multiply(diff, diff))
    return rss

def inverse_regression_predictions(output, intercept, slope):
    return (output - intercept)/slope

# Load the King County house data

# Use the type dictionary supplied
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

# Split data into 80% training and 20% test - But this is a random split so we aren't going to use it for the exercise
#train_data, test_data = cv.train_test_split(df, test_size = 0.2)
#input_feature = train_data['sqft_living']
#output = train_data['price']

# Using the specific training data
input_feature1 = train['sqft_living']
output = train['price']

squarefeet_intercept, squarfeet_slope = simple_linear_regression(input_feature1, output)
y_hat = get_regression_predictions(2650.0, squarefeet_intercept, squarfeet_slope)
print("Predicted price: ", y_hat)

rss_squarefeet = get_residual_sum_of_squares(input_feature1, output, squarefeet_intercept, squarfeet_slope)
print("RSS_squarefeet on training data: ", rss_squarefeet)

y_hat = 800000
x = inverse_regression_predictions(y_hat, squarefeet_intercept, squarfeet_slope)
print("House area: ", x)

input_feature2 = train['bedrooms']
bedroom_intercept, bedroom_slope = simple_linear_regression(input_feature2, output)
rss_bedroom = get_residual_sum_of_squares(input_feature2, output, bedroom_intercept, bedroom_slope)
print("RSS_bedroom on training data: ", rss_bedroom)

# Computing RSS on test data
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
input_feature1 = test['sqft_living']
output = test['price']
squarefeet_intercept, squarfeet_slope = simple_linear_regression(input_feature1, output)
rss_squarefeet = get_residual_sum_of_squares(input_feature1, output, squarefeet_intercept, squarfeet_slope)
print("RSS_squarefeet on test data: ", rss_squarefeet)

input_feature2 = test['bedrooms']
bedroom_intercept, bedroom_slope = simple_linear_regression(input_feature2, output)
rss_bedroom = get_residual_sum_of_squares(input_feature2, output, bedroom_intercept, bedroom_slope)
print("RSS_bedroom on test data: ", rss_bedroom)



