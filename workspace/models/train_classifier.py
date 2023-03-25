import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import joblib
import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(database_filepath):
    '''
    load_data
    Loads data stored in a database file into the X (feature)
    and y (label) variables. The column names are stored in another variable.

    INPUT:

    database_filepath - File location for the database

    OUTPUT:

    X - Feature Dataframe
    Y - Label Dataframe
    categroy_names - column names in the dataframe made from the database

    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    y = df.drop(columns=df.columns[0:4], axis=1)
    category_names = df.columns

    return X, y, category_names


def tokenize(text):
    '''
    tokenize
    Filters the text variable so that it:
        1. Only includes letters and numbers.
        2. Every letter is lower case.
        3. Tokenize the text.
        4. Lemmatize the text.
        5. Strips the text of white space.
        6. Removes stop words.

    INPUT:

    text - string to be filtered.

    OUTPUT:

    tokens - filtered text that has been tokenized.

    '''
    # Removes everything except letters and numbers. Converts to lower case.
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    # Lemmatizes, Strips, and removes stopwords from text.
    tokens = [WordNetLemmatizer().lemmatize(w).strip()
              for w in tokens if w not in stopwords.words("english")]

    return tokens


def build_model():
    '''
    build_model
    Builds the pipeline using:
        1. Count Vectorizer
        2. Tfidf Transformer
        3. Multioutput Classifier with a Logistic Regression estimator.
    A grid search is used to see what would be the best C value.

    OUTPUT:

    cv - optimized pipeline.

    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000)))
        ])

    parameters = {
        'clf__estimator__C': [0.1, 1.0]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Creates the prediction variable, and then runs a classification report to
    display the precision, recall, and f1-score of each label.

    INPUT:

    model - optimized pipeline
    X_test - test feature data
    Y_test - test label data
    category_names - names of the labels

    '''
    y_pred = model.predict(X_test)

    for col in range(y_pred.shape[1]):

        print('Label Name: ' + category_names[col])
        print(classification_report(Y_test.iloc[:, col], y_pred[:, col],
                                    labels=np.unique(y_pred[:, col])))


def save_model(model, model_filepath):
    '''
    save_model
    Saves the optimized model in a specified location as a pickle file.
    '''
    joblib.dump(model, model_filepath)


def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label


def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub


def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=5)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbor,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...\n')
        evaluate_model(model, X_test, Y_test, category_names[4:])

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
