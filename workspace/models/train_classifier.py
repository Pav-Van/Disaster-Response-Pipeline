import sys
import pandas as pd
import numpy as np
import scipy as sp
import re
from sqlalchemy import create_engine
import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

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
        ('clf', MultiOutputClassifier(estimator=DecisionTreeClassifier()))
        ])

    parameters = {
        "clf__estimator__criterion": ['gini', 'entropy']
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
