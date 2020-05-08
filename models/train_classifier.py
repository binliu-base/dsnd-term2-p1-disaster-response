import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, \
    f1_score, make_scorer, precision_score


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization


def load_data(database_filepath):
    """ Loads X, Y and gets category names
        Args:
            database_filepath (str): string filepath of the sqlite database
        Returns:
            X (pandas dataframe): Feature data, just the messages
            Y (pandas dataframe): labels
            category_names (list): List of the category names for classification
    """ 
    # table name
    my_table = 'clean_df'
    engine = create_engine('sqlite:///' + database_filepath)    
    df = pd.read_sql_table(my_table, engine) 
    X = df['message']
    Y = df.iloc[:, 4:] 
    return X, Y, Y.columns


def tokenize(text):
    """ Tokenize the text.
        Args: 
            text
        Returns:
            Tokenized text
    """       
    # remove special characters and lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  
    
    # tokenize
    tokens = word_tokenize(text)  
    
    # lemmatize, remove stopwords   
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """    

    # PRELIMINARY PIPELINE
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer = tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
                        ])

    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0)
    }


    return GridSearchCV(pipeline, param_grid=parameters)    

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        None
    """
    y_pred = model.predict(X_test)

    # for i in range(len(category_names)):
    #     accuracy = accuracy_score(Y_test.iloc[:, i], Y_pred[:, i],average='weighted')
    #     precision = precision_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted')
    #     recall = recall_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted')
    #     f1 = f1_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted')
    #     print("category: {},  accuracy={:.2f}, precision={:.2f}, recall={:.2f}, f1_score={:.2f}".format(category_names[i], accuracy, precision, recall, f1))

    for i, col  in enumerate(category_names):
        ytrue = Y_test[col]
        ypred = y_pred[:,i]
        print(col)
        print(classification_report(ytrue, ypred))        
        print('-' * 60)    




def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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
