##########library list###############
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class LGModel(object):

    def __init__(self):
        """
            Attributes:
            clf: default sklearn LogisticRegression classifier model
            vectorizor: TFIDF vectorizer with 1-3 grams, 5 minimal document frequency, and used idf weights
            Ref:https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.score
        """
        self.clf = LogisticRegression(multi_class = 'auto')
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 5,stop_words='english', use_idf=True)

    def vectorizer_fit(self, X):
        """
            Fits a tfidf vectorizer to the text
        """
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """
            Transform the text data to a tfidf matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """
            Trains the classifier with the matrix and target class
        """
        self.clf.fit(X, y)

    def predict(self, X):
        """
            Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def predict_proba(self, X):
        """
            Returns probability of predictions
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba

    def performance(self,X,y):
        """
            return the accuracy_score of the classifer
        """
        return  self.clf.score(X, y)

    def pickle_vectorizer(self, path='models/Vectorizer'):
        """
            Saves the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='models/lg_clf'):
        """
            Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))
