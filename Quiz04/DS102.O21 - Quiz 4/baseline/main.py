import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

# Defines a function for training the model.


def run_train(public_dir, model_dir):
    # Ensures the model directory exists, creates it if it doesn't.
    os.makedirs(model_dir, exist_ok=True)

    # Constructs the path to the training data file.
    train_file = os.path.join(public_dir, 'train.npz')

    # Loads the training data from the .npz file.
    train_data = np.load(train_file)

    # Extracts the features from the training data.
    X_train = train_data['X_train'][:,:60]

    # Extracts the labels from the training data.
    y_train = train_data['y_train']

    # Instantiates the Gradient Boosting model.
    # model = LogisticRegression(
    #     C=0.155221171236024, penalty='l1', solver='liblinear')
    # model = SVC(
    # C=0.5, kernel='rbf', gamma='auto')

    # model = DecisionTreeClassifier()
    # criterion='entropy', max_depth=18, max_features=None, min_samples_leaf=1, min_samples_split=5)
    model = KNeighborsClassifier(
    n_neighbors=17, weights='distance')
    # model = GaussianNB()
    # model = RandomForestClassifier()
    # model = GradientBoostingClassifier(
    # max_depth=9, max_features='sqrt', min_samples_leaf=6, min_samples_split=6, n_estimators=236)
    # model = HistGradientBoostingClassifier()
    # max_bins=34, max_depth=9, max_iter=506, min_samples_leaf=8)
    # model = AdaBoostClassifier()
    # base_estimator=DecisionTreeClassifier(
    #     criterion='entropy', max_depth=18, max_features=None, min_samples_leaf=1, min_samples_split=5), n_estimators=100)
    # model = xgb.XGBClassifier(
    # booster='gblinear', eta=0.2, gamma=0.1, max_depth=9, min_child_weight=1, n_estimators=100, subsample=0.8)
    # model = lgb.LGBMClassifier()
    # boosting_type='gbdt', learning_rate=0.1, max_depth=9, n_estimators=100, num_leaves=31, objective='binary')

    # Fits the model to the training data.
    model.fit(X_train, y_train)

    # Defines the path for saving the trained model.
    model_path = os.path.join(model_dir, 'trained_model.joblib')

    # Saves the trained model to the specified path.
    dump(model, model_path)


# Defines a function for making predictions.
def run_predict(model_dir, public_dir, output_path):
    # Specifies the path to the trained model.
    model_path = os.path.join(model_dir, 'trained_model.joblib')

    # Constructs the path to the test data file.
    test_file = os.path.join(public_dir, 'test.npz')

    # Loads the trained model from file.
    model = load(model_path)

    # Loads the test data from the .npz file.
    test_data = np.load(test_file)

    # Extracts the features from the test data.
    X_test = test_data['X_test'][:,:60]

    # Predicts and saves results.
    pd.DataFrame({'y': model.predict(X_test)}).to_json(
        output_path, orient='records', lines=True)


# Defines the main function that parses commands and arguments.
def main():
    # Initializes a parser for command-line arguments.
    parser = argparse.ArgumentParser()

    # Creates subparsers for different commands.
    subparsers = parser.add_subparsers(dest='command')

    # Adds a subparser for the 'train' command.
    parser_train = subparsers.add_parser('train')

    # Adds an argument for the directory containing public data.
    parser_train.add_argument('--public_dir', type=str)

    # Adds an argument for the directory to save the model.
    parser_train.add_argument('--model_dir', type=str)

    # Adds a subparser for the 'predict' command.
    parser_predict = subparsers.add_parser('predict')

    # Adds an argument for the directory containing the model.
    parser_predict.add_argument('--model_dir', type=str)

    # Adds an argument for the directory containing public data.
    parser_predict.add_argument('--public_dir', type=str)

    # Adds an argument for the path to save prediction results.
    parser_predict.add_argument('--output_path', type=str)

    # Parses the command-line arguments.
    args = parser.parse_args()

    if args.command == 'train':
        # Checks if the 'train' command was given.
        # Calls the function to train the model.
        run_train(args.public_dir, args.model_dir)
    elif args.command == 'predict':
        # Checks if the 'predict' command was given.
        # Calls the function to make predictions.
        run_predict(args.model_dir, args.public_dir, args.output_path)
    else:
        # If no valid command was given, prints the help message.
        # Displays help message for the CLI.
        parser.print_help()

        # Exits the script with a status code indicating an error.
        sys.exit(1)


# Checks if the script is being run as the main program.
if __name__ == "__main__":
    # Calls the main function if the script is executed directly.
    main()
