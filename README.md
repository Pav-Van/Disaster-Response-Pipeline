# Disaster Response Pipeline Project

## Installation
 
The Anaconda distribution of Python is needed to run the code, no additional libraries are needed. The notebook uses python 3.10.9

## Project Description and Motivation

This project classifies messages typically transmitted because of emergency situations. There are thirty five different categories 
the message could be classified as. The dataset supplied by Appen was used to train a machine learning pipeline that uses a 
logistic regression classifier. When this project is run, it builds a web application that allows the user to enter a message.
Once the message is entered, the user can hit the "Classify Message" button to see how the message was classified. If there is
a "1" next to any category, it means the message was classified as relating to that category. The web app also shows some visualizations
that describe the training data. This web app was built to help emergency response teams filter and organize the many messages that
are recieved during emergency situations.

## File Descriptions

* Juptyer Notebooks
    * categories.csv # data to process
    * DisasterResponse.db # database to save clean data to
    * ETL Pipeline Preperation.ipynb # Preliminary ETL pipeline code
    * messages.csv # data to process
    * ML Pipeline Preparation.ipynb # Preliminary ML pipeline code
    * Pipeline_log.pkl # Preliminary saved model
    
* workspace
    * app
        * templates
            * go.html # classification result page of web app
            * master.html # main page of web app
        * run.py # Flask file that runs app
    * data
        * disaster_categories.csv # data to process
        * disaster_messages.csv # data to process
        * DisasterResponse.db # database to save clean data to
        * process_data.py # Python script that cleans data
    * models
        * classifier.pkl # saved model
        * train_classifier.py # Python script that creates the classifier
        
* .gitignore # holds filenames that are not tracked by Git
* LICENSE.md # MIT License information
* README.md

## Results

The classifer I used (Logistic Regression) does an average job at categorizing the messages. There is still alot of room for improvement. Testing
different classifiers would probably help the accuracy of the classification. Based on the heatmap, it looks like a vast majority of the categories do not correlate with each other. The "related" category was used the most in the training data, followed by the "aid_related" and "weather_related" categories

## Licensing, Authors, Acknowledgements

The credit for the data goes to Appen. Also credit to Udacity for the detailed and helpful lessons given online to help me create this notebook. This code is released under the MIT License.

### Instructions:
1. In the workspace directory, run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
