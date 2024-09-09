from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd
import datetime
import os
import pickle
from nltk.stem import WordNetLemmatizer
import nltk

class LemmaTokenizer:
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wordnetlemma.lemmatize(token) for token in nltk.word_tokenize(doc)]

app = Flask(__name__)
#path = r'C:\Users\N THANUSH\Desktop\AB\classifier.pkl'
#a=pickle.load(open("classifier.pkl","rb"))
a = joblib.load("me.pkl")

def predictfunc(review):     
     prediction = a.predict(review)
     if prediction[0]=='positive':
          sentiment='Positive'
          prediction[0]=1
     else:
          sentiment='Negative'    
          prediction[0]=0  
     return prediction[0],sentiment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
     
     if request.method == 'POST':
        result = request.form
        content = request.form['review']
        review = pd.Series(content)
        prediction,sentiment =predictfunc(review)      
     return render_template("review.html",pred=prediction,sent=sentiment)

if __name__ == '__main__':
     app.run(port=5500,debug = True)
     #app.run(host='0.0.0.0')