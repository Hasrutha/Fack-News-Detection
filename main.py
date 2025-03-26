from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Define stemming function
nltk.download('stopwords')
ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train model
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')  # Create a simple HTML form in "templates/index.html"

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news_text']
    input_data = vectorizer.transform([input_text])
    prediction = model.predict(input_data)
    
    result = "Fake News" if prediction[0] == 1 else "Real News"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)