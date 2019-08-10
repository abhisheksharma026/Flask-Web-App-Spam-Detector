from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['Label'].map({'ham': 0, 'spam': 1})
	X = df['Message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer(max_features = 3000)
	X = cv.fit_transform(X) # Fit the Data

	#Usage of Saved Model
	NB_spam_model = open('NB_spam_model.pkl','rb')
	clf = joblib.load(NB_spam_model)
    
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(port = 8000, debug=True)
