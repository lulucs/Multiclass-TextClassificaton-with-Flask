##########library list###############
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from model import LGModel
import re
import pickle
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



def clean_text(text):
    """
        text: a string
        return: cleaned initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = re.compile('[/(){}\[\]\|@,;]').sub(' ', text) # replace matched symbols by space in text
    text = re.compile('[^0-9a-z #+_]').sub('', text) # delete symbols which are in symbols_re from text
    text = text.lower()
    return text

def labelid(df):
    """
        convert categorical label into numbers and buid corresponding dictionary
    """
    df['cat_id'] = df.tags.factorize()[0]
    category_id_df = df[['tags', 'cat_id']].drop_duplicates().sort_values('cat_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['cat_id', 'tags']].values)
    return[df,id_to_category]

def build_model():
    """
        train a classifier with stackoverflow data
    """
    #load data
    model = LGModel()
    print("=========loading data===========")
    url = "https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv"
    df = pd.read_csv(url)

    #get a subset of the data
    print("=========preprocessing data===========")
    categories = ['javascript', 'python', 'css', 'mysql', 'iphone', 'html', 'ios', 'php']
    df=df[df.tags.isin(categories)]

    #clean HTML-formated data
    df['post'] = df['post'].apply(clean_text)

    #encode target class and save dictionary
    df, id_to_category = labelid(df)
    with open("models/dict",'wb') as f:
        pickle.dump(id_to_category,f)

    #convert data into tdm
    print("=========construct tdm ==========")
    model.vectorizer_fit(df.post)
    X = model.vectorizer_transform(df.post)
    y = df.cat_id

    #train the classifier
    print("=========learning model===========")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1111)
    model.train(X_train, y_train)
    model.pickle_clf()
    model.pickle_vectorizer()
    print("=========I'm the model =D and here is my performance===========")

    # evaluate the model
    y_pred = model.clf.predict(X_test)
    ## display the performance
    print("Model accuracy score: "+ str(model.performance(X_test, y_test)))
    print(classification_report(y_test, y_pred,target_names=categories))

if __name__ == '__main__':
    build_model()
