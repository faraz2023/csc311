import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, max_error, mean_absolute_error
import seaborn as sns

np.random.seed(42)

# this helper function normazlies both vector and matrices through method l2
def general_normalizer(X, vector=False):
    if vector:
        return preprocessing.normalize(X[:,np.newaxis], axis=0).ravel()
    return preprocessing.normalize(X, axis=0)

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features

# loads design, target matrices into a df
def load_as_df(X, y, features):
    df = pd.DataFrame(X, columns=features)
    df['TARGET'] = y
    return df


def visualize(df):
    plt.figure(figsize=(20, 5))

    data_features = df.columns[:-1]


    # i: index
    i = 0
    for feature in (data_features):
        p = plt.subplot(3, 5, i + 1)
        i += 1
        p.scatter(df[feature], df['TARGET'], s=0.22)
        p.set_xlabel('Normalized ' + feature)
        p.set_ylabel('Target')
        p.set_title("Target to " + feature)

    plt.tight_layout()
    plt.show()

# helper function to give us summary statistics
def summarize_and_describe(df):
    print(df.describe())
    print('--------')

    print("Features' Names: {}".format(df.columns[:-1]))
    print('--------')

    print("Head of the DataFrame:")
    print(df.head())
    print('--------')

    print("Dimensions of the Data: ")
    print("\t", end='')
    print(df.shape)
    print('--------')


    print("Number of Data Points: ")
    print("\t", end='')
    print(df.shape[0])
    print('--------')



def fit_regression(X, Y):

    X_transposed = X.transpose()

    #solving the equastion from part (a)
    left_side = np.matmul(X_transposed, X)
    right_side = np.matmul(X_transposed, Y)

    return np.linalg.solve(left_side, right_side).flatten()

def predict(weights, X):
    return np.matmul(X, weights)

#prints the intercept and weights in human readable format
def tabulate_model(weights, features):
    features = list(features)
    features = ['Intercept'] + features
    table_dict = dict(zip(features, weights))
    print("{:<15} {:<10}".format('Feature', 'Weight'))
    print("-------         ------")
    for v in table_dict.items():
        feature, weight = v
        print("{:<15} {:.10f}".format(feature, weight))

# computes in different ways the accuracy of our model
def messuare_errors(y_test, y_pred):
    # Mean Squared Error
    print('Mean squared error: %.9f'
          % mean_squared_error(y_test, y_pred))
    # The Mean Absolute Error
    print('Mean absolute  error: %.9f'
          % mean_absolute_error(y_test, y_pred))
    # Max Error
    print('Max error: %.9f'
          % max_error(y_test, y_pred))

    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))


def main():
    # Load the data
    X, y, features = load_data()

    # adding bias term
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    #normalize data
    X = general_normalizer(X)
    y = general_normalizer(y, vector=True)

    #load data as dataframe (used for visualization, descriptive statistics)
    #we get rid of the bias term
    df = load_as_df(X[:,1:], y, features)

    # Visualize the features
    #visualize(df)

    #summarize and describe
    #summarize_and_describe(df)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


    # Fit regression model bias term included (returns: d+1 vector)
    w = fit_regression(X_train, y_train)

    #Tabulated weights
    print("Tabulated weights: ")
    tabulate_model(w, features)

    # predict the testing set
    y_pred = predict(w, X_test)

    messuare_errors(y_test, y_pred)




if __name__ == "__main__":
    main()

