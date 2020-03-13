import os
import argparse
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
import prepare_dataset


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

def load_datasets(args):

    # load the dataset
    #df17 = pd.read_csv(utils.loc / 'data' / 'data17_sample100000.csv')
    df17 = prepare_dataset.sample('17', args.sample)
    df18 = prepare_dataset.sample('18', args.sample)
    #df18 = pd.read_csv(utils.loc / 'data' / 'data18_sample100000.csv')

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

    return m_train, m_valid


def train_model(args, outloc):

    print('Training model {}'.format(args.name))

    # output location
    if outloc.exists():
        if args.force:
            os.system('rm -r {}'.format(outloc))
        else:
            print('Model with name {} already exists, please choose another model or use the --force to overwrite.'.format(args.name))
            exit()
    outloc.mkdir(parents=True)

    # load dataset
    m_train, m_valid = load_datasets(args)

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

    # save the model
    model.save_model(str(outloc / 'model.xgb'))
    return model


def evaluate_model(args, outloc):

    # get the model
    try:
        print('Loading model {}'.format(args.name))
        model = xgboost.Booster()
        model.load_model(str(outloc / 'model.xgb'))
    except xgboost.core.XGBoostError:
        print('No such model exists, training it from scratch.')
        model = train_model(args, outloc)

    # get the dataset
    m_train, m_valid = load_datasets(args)
    Y_train = m_train.get_label()
    Y_valid = m_valid.get_label()

    # compute the baseline loss when using the dataset mean
    mean_preds = Y_train.mean() * np.ones(Y_train.shape)
    train_error = mean_absolute_error(Y_train, mean_preds)
    valid_error = mean_absolute_error(Y_valid, mean_preds)

    # compute the losses throughout the training 
    print('Computing predictions for partial models (first x trees)')
    num_trees = len(model.get_dump())
    val_loss = []
    train_loss = []
    trees = np.array([t for t in range(1, num_trees)])
    for t in trees:
        # validation
        y_pred = model.predict(m_valid, ntree_limit=t)
        val_loss.append(mean_absolute_error(Y_valid, y_pred))
        # training
        y_pred = model.predict(m_train, ntree_limit=t)
        train_loss.append(mean_absolute_error(Y_train, y_pred))

    # create the training plot
    fig, ax = plt.subplots()
    ax.plot(trees, val_loss, color='C0', label='validation loss')
    ax.plot(trees, train_loss, color='C1', label='train loss')
    ax.plot(trees, valid_error*np.ones(trees.shape), color='C0', linestyle=':', label='baseline valid')
    ax.plot(trees, train_error*np.ones(trees.shape), color='C1', linestyle=':', label='baseline train')
    ax.set_ylim(0, 0.15)
    ax.legend()
    ax.set_xlabel('Training stage')
    ax.set_ylabel('Mean absolute error')
    plt.savefig(outloc / 'LossHistory.pdf')


def main():

    # parse the arguments
    parser = argparse.ArgumentParser(description='prepare the dataset')
    parser.add_argument('--name', type=str, default='default',
                        help='Give the run a special name')
    parser.add_argument('--sample', type=int, default=100000,
                        help='Number of samples from the full dataset')
    parser.add_argument('--force', default=False, action='store_true',
                        help='overwrite the output file')
    parser.add_argument('--train', default=False, action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', default=False, action='store_true',
                        help='evaluate the model')
    args = parser.parse_args()

    # model directory
    outloc = utils.loc / 'data' / 'model_{}_{}'.format(args.name, args.sample)

    # train
    if args.train:
        train_model(args, outloc)

    if args.evaluate:
        evaluate_model(args, outloc)


if __name__ == '__main__':
    main()
