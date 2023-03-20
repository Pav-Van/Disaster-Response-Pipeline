import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads data stored in .csv files into the messages and category
    dataframes. These variables are then merged into a single single dataframe.
    
    INPUT:

    messages_filepath - File location for the messages .csv file
    categories_filepath - File location for the categories .csv file

    OUTPUT:

    df - Merged dataframe containing both the messages and categorys data
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=('id'))
    return df


def clean_data(df):
    '''
    This function cleans and returns a dataframe.
    
    INPUT:

    df - Dataframe to be cleaned

    OUTPUT:

    df - Cleaned Dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2]).to_list()
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # columns that have only one value
    drop_labels = categories.loc[:,((categories.mean() == 0) | (categories.mean() == 1))].columns.tolist()
    
    # drop the columns that only have one value 
    categories.drop(labels=drop_labels,axis=1, inplace=True)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
       
    return df

def save_data(df, database_filename):
    '''
    This function saves the data stored in the df variable into a SQL database.
    
    INPUT:

    df - Dataframe storing all the data
    database_filepath - File location for newley created database

    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
       

def main():
        
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()