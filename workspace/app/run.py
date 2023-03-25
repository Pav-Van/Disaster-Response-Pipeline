import json
import plotly
import pandas as pd
import joblib
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter, Heatmap
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    '''
    This function Tokenizes strings. Lemmatization, lower, and strip functions
    are used to clean the text.

    INPUT:

    text - string to be filtered.

    OUTPUT:

    tokens - filtered text that has been tokenized.

    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # selecting the sum of the last 36 columns, sorting them in descending order, and splicing the top 10.
    category_proportion = ((df.iloc[:, 4:].sum())/df.shape[0]).sort_values(ascending=False)[:10]
    category_names = list(category_proportion.index)

    # creating correlation matrix for heatmap, dropped the first four columns
    df_corr = df.iloc[:, 4:].corr()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
            }
        },
        {
            'data': [
                Heatmap(
                    x=df_corr.columns,
                    y=df_corr.index,
                    z=np.array(df_corr)
                )
            ],

            'layout': {
                'title': 'Category Heatmap',
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_proportion
                )
            ],

            'layout': {
                'title': 'Top 10 Selected Categories',
                'yaxis': {
                    'title': "Proportion of Categories Selected"
                },
                'xaxis': {
                    'title': {
                        'text': "Category Name",
                    }
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
