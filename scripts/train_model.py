import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')
import fashion.preprocessing as prep
from fashion import utils



def prepare_X(X):

    # turn booleans into integers
    for feature in X.columns:
        dtype = X.dtypes[feature]
        if dtype == np.bool:
            X[feature] = X[feature].astype(int)

    # and one-hot encode categorical variables
    X = pd.get_dummies(X)

    return X


def main():

    print('Hello world')

    # load the dataset
    df = pd.read_csv(utils.loc / 'data' / 'basic_2017.csv')

    # define the features and the target
    features = ['Week', 'Franchise', 'Gender', 'Season', 'OriginalListedPrice']
    target = 'Volume'
    X = df[features]
    Y = df[target]

    # prepare the dataset
    X = prepare_X(X)

    # split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75)

    # prepare the model
    n_estimators = 100
    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                      verbose=1)
    model.fit(X_train, Y_train)

    # plot the training
    stages = range(n_estimators)
    val_loss = []
    train_loss = []
    for y_pred in model.staged_predict(X_test):
        val_loss.append(model.loss_(Y_test, y_pred))
    for y_pred in model.staged_predict(X_train):
        train_loss.append(model.loss_(Y_train, y_pred))
    
    fig, ax = plt.subplots()
    ax.plot(stages, val_loss, label='validation loss')
    ax.plot(stages, train_loss, label='train loss')
    ax.legend()
    ax.set_xlabel('Training stage')
    ax.set_ylabel('Loss')
    plt.savefig(utils.loc / 'figures' / 'training' / 'BaselineTraining.pdf')


if __name__ == '__main__':
    main()
