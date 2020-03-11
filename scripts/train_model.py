import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
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

    # preprocess each feature separately
    features = list(X.columns)
    for f in features:

        dt = X.dtypes[f]

        # let numericals pass
        if dt in [np.float64, np.int64]:
            continue

        # and encode categoricals
        else:
            # depending on the number of unique elements
            n_unique = len(X[f].unique())

            # splurge with memory
            if n_unique < 10:
                new = pd.get_dummies(X[f], prefix=f)
                X = pd.concat([X, new], axis=1)
                X.drop(f, axis=1, inplace=True)

            # or be stingy
            else:
                X[f] = pd.Categorical(X[f]).codes

    return X


def main():

    # load the dataset
    df17 = pd.read_csv(utils.loc / 'data' / 'data17_sample100000.csv')
    df18 = pd.read_csv(utils.loc / 'data' / 'data18_sample100000.csv')

    # define the features and the target
    features = ['Week', 'Franchise', 'Gender', 'Season', 'OriginalListedPrice']
    target = 'Volume'

    # and add the features that were computed rather than directly available
    features += ['NUniqueProductsSold', 'NTotalProductsSold']

    # prepare the train and test parts
    X_train = prepare_X(df17[features])
    Y_train = df17[target]
    m_train = xgboost.DMatrix(X_train, label=Y_train)

    X_valid = prepare_X(df18[features])
    Y_valid = df18[target]
    m_valid = xgboost.DMatrix(X_valid, label=Y_valid)

    # prepare the model
    num_round = 1000
    params = {'max_depth': 4,
              'eta': 0.1,
              'objective':'reg:squarederror',
              'eval_metric':'mae', 
              'colsample_bytree':0.05,
              'tree_method':'hist'
              }

    model = xgboost.train(params,
                          m_train,
                          num_round,
                          evals=[(m_valid, 'valid')],
                          early_stopping_rounds=5,
                          #verbose_eval=100)
                          )
    print(model.attributes())

    # compute the losses throughout the training 
    print('Computing predictions for partial models (first x trees)')
    num_trees = len(model.get_dump())
    val_loss = []
    train_loss = []
    trees = range(1, num_trees)
    for t in trees:
        # validation
        y_pred = model.predict(m_valid, ntree_limit=t)
        val_loss.append(mean_absolute_error(Y_valid, y_pred))
        # training
        y_pred = model.predict(m_train, ntree_limit=t)
        train_loss.append(mean_absolute_error(Y_train, y_pred))

    # create the training plot
    fig, ax = plt.subplots()
    ax.plot(trees, val_loss, label='validation loss')
    ax.plot(trees, train_loss, label='train loss')
    ax.legend()
    ax.set_xlabel('Training stage')
    ax.set_ylabel('Mean absolute error')
    plt.savefig(utils.loc / 'figures' / 'training' / 'BaselineTraining.pdf')


if __name__ == '__main__':
    main()
