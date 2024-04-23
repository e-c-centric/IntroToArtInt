from flask import Flask, render_template, request
import random
import time
import pyttsx3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import textstat
from gensim.models import Word2Vec
import joblib

app = Flask(__name__)

kmeans = joblib.load('kmeans_model.pkl')
application_df = pd.read_csv('application_words.txt', sep=" ", names=['word'])
stored_clusters = []
score = 0

word2vec_model = Word2Vec(sentences=[application_df['word'].tolist()], vector_size=100, window=5, min_count=1, workers=4)
def word_to_vec(word):
    try:
        return word2vec_model.wv[word]
    except KeyError:
        return None
    
def play_audio(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()

def extract_features(word):
    try:
        syllables = textstat.syllable_count(word)
        ari = textstat.automated_readability_index(word)
        length = len(word)
        #language, _ = langid.classify(word)

        return {'syllables': syllables, 'ari': ari, 'length': length, 'language': None}
    except Exception as e:
        print(f"Error extracting features for '{word}': {e}")
        return None

def predict_cluster(word):
    word_vector = word_to_vec(word)
    features = extract_features(word)
    ari = features['ari']
    length = features['length']
    #lang = features['language']
    features=[ari,length]
    if None in features or word_vector is None:
        print(f"Error extracting features for '{word}'. Unable to predict the cluster.")
        return None
    input_features = np.concatenate([word_vector, features])
    input_features = input_features.reshape(1, -1)
    predicted_cluster = kmeans.predict(input_features)
    return predicted_cluster[0]

@app.route('/')
def index():
    global application_df
    global stored_clusters
    global score

    if not application_df.empty:
        random_words = application_df['word'].sample(n=4)
        random_words = random_words.tolist()
        index = random.randint(0, 3)
        word_to_pronounce = random_words[index]

        play_audio(word_to_pronounce)
        
        return render_template('index.html', word=word_to_pronounce, options=random_words)
    
    return render_template('game_over.html', score=score)

@app.route('/answer', methods=['POST'])
def answer():
    global stored_clusters
    global score

    user_input = int(request.form['answer'])
    word_to_pronounce = request.form['word']
    
    if user_input == stored_clusters.index(word_to_pronounce) + 1:
        stored_clusters = []
        combo_bonus = len(stored_clusters) * 10
        score += 10 + combo_bonus

    return render_template('answer.html', is_correct=user_input == stored_clusters.index(word_to_pronounce) + 1, word=word_to_pronounce, score=score)

if __name__ == '__main__':
    app.run(debug=True)
