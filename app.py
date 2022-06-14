from flask import Flask,render_template,url_for,request
import pickle

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

app = Flask(__name__)

filename = "finalized_model.pkl"
loaded_model = pickle.load(open(filename, 'rb'))

def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in words:
        features[word] = (word in words)

    return features

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		comment = request.form['comment']
		komentar = find_features(comment)
		my_prediction = loaded_model.classify(komentar)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)