# app.py
from flask import Flask, render_template, request
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Load the model, vectorizer, and dataset
model = joblib.load('./model.pkl')
vectorizer = joblib.load('./vectorizer.pkl')
dataset = pd.read_csv('./disease_management_dataset.csv')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Preprocessing function
def remove_punc(text):
    return re.sub(r'[^\w\s]', '', text).lower()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to preprocess the text input
def preprocess_text(text):
    text = remove_punc(text)
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    custom_lemmas = {
        "runny": "run", "wheezing": "wheeze", "tightness": "tight",
        "sneezing": "sneeze", "sometimes": "sometime"
    }

    cleaned_words = [
        custom_lemmas.get(word.lower(), lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos)))
        for word, pos in pos_tags if word.lower() not in stop_words
    ]

    return ' '.join(cleaned_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptom_description = request.form['symptom_description']
        processed_text = preprocess_text(symptom_description)
        symptoms_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(symptoms_vectorized)[0]

        # Find the row in the dataset that matches the predicted disease
        disease_info = dataset[dataset['disease'] == prediction]
        if not disease_info.empty:
            # Extract the details for the matched disease
            precautions = disease_info['precautions'].values[0]
            medications = disease_info['medications'].values[0]
            workout = disease_info['workout'].values[0]
            diet = disease_info['diet'].values[0]
        else:
            # Default values if disease not found in dataset
            precautions = medications = workout = diet = "Information not available"

        return render_template(
            'index.html',
            prediction=prediction,
            precautions=precautions,
            medications=medications,
            workout=workout,
            diet=diet
        )

if __name__ == '__main__':
    app.run(debug=True)
