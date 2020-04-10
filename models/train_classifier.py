import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
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
    """ A series of nested functions to preprocess the text data
        Functions: 
            Tokenize Text 
            remove special characters
            lemmatize_text
            remove_stopwords
        Returns:
            Clean and preprocessed text 
    """       
    default_stopwords = set(stopwords.words("english"))  

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip()  ) 

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)] 

    def remove_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))        

    def lemmatize_text(text):
        tokens = tokenize_text(text)
        lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens]
        # Lemmatize verbs by specifying pos
        lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed] 
        # Reduce words to their stems
        stemmed = [PorterStemmer().stem(w) for w in text]

        return stemmed

    text = remove_characters(text) # remove punctuation and symbols
    tokens = lemmatize_text(text) # stemming

    # Remove stop words
    words = [w for w in tokens if w not in default_stopwords]        
    
    return ' '.join(tokens)


def build_model():
    """Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """    
    forest = RandomForestClassifier(n_estimators = 10, random_state = 2) 

    # PRELIMINARY PIPELINE
    pipeline = Pipeline([
                         ('cvect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(forest))
                         ])

    parameters = {
        'cvect__min_df': [2, 4],
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators':[10, 50],
        'clf__estimator__min_samples_split':[2, 4, 8]
    }

    # parameters = {
    #     'cvect__min_df': [2],
    #     'tfidf__use_idf':[False],
    #     'clf__estimator__n_estimators':[10],
    #     'clf__estimator__min_samples_split':[2]
    # }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs= 1)    

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
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test.iloc[:, i], Y_pred[:, i])
        precision = precision_score(Y_test.iloc[:, i], Y_pred[:, i])
        recall = recall_score(Y_test.iloc[:, i], Y_pred[:, i])
        f1 = f1_score(Y_test.iloc[:, i], Y_pred[:, i])
        print("category: {},  accuracy={:.2f}, precision={:.2f}, recall={:.2f}, f1_score={:.2f}".format(category_names[i], accuracy, precision, recall, f1))


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
