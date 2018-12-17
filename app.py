##########main app###############
#library list#
from flask import Flask
from flask_restful import reqparse, Api, Resource
import pickle
import numpy as np
from model import LGModel
from build_model import clean_text

#initialize app and model#
app = Flask(__name__)
api = Api(app)
model = LGModel()

#load the model, vectorizer and dictionary#
with open('models/lg_clf', 'rb') as f:
    model.clf = pickle.load(f)
with open('models/Vectorizer', 'rb') as f:
    model.vectorizer = pickle.load(f)
id_to_category={}
with open("models/dict",'rb') as f:
    id_to_category = pickle.load(f)




# argument parsing for user input#
parser = reqparse.RequestParser()
parser.add_argument('query')

#prediction function block#
class PredictCategory(Resource):
    def get(self):
        # use parser and get the user's query
        args = parser.parse_args()
        user_query = args['query']
        user_query = clean_text(user_query)

        # vectorize the user's query and make a prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)
        pred_text = id_to_category[prediction[0]]

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[:,prediction[0]], 3)

        # create JSON object
        output = {'confidence': confidence,'prediction': pred_text}
        return output

# Setup the Api resource routing
# Route the URL to the resource
api.add_resource(PredictCategory, '/')

if __name__ == '__main__':
    app.run(debug=True)
