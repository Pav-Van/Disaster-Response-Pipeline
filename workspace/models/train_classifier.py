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
from sklearn.linear_model import LogisticRegression

nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df.message.values
    y = df.drop(columns=df.columns[0:4],axis=1)
    category_names = df.columns
    
    return X,y,category_names


def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)

    tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in tokens if w not in stopwords.words("english")]

    return tokens


def build_model(X_train, Y_train):

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000)))
    ]) 

    print('Training model...')
    pipeline.fit(X_train, Y_train)

    parameters = {
    'clf__estimator__C': [0.1, 1.0]    
    }

    cv = GridSearchCV(pipeline,param_grid=parameters)

    cv.fit(X_train,Y_train)
  
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000,C=cv.best_params_['clf__estimator__C'])))
    ]) 

    pipeline.fit(X_train, Y_train)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)

    for col in range(y_pred.shape[1]):

        print(classification_report(Y_test.iloc[:,col],y_pred[:,col],labels=np.unique(y_pred[:,col])))
        print(category_names[col])
        print(col)
      


def save_model(model, model_filepath):

    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()