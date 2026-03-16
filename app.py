import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model


app = Flask(__name__)

model = load_model('models/next_word_lstm.h5')
with open('models/tokenizer.pickel', 'rb') as f:
    tokenizer = pickle.load(f)


def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len -1):] # ensure this seq lenth is match max_seq
    token_list = pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
    predicted = model.predict(token_list)
    predict_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predict_word_index:
            return word
    return None
@app.route('/')
def index():
    return render_template('predict.html', next_word='')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        max_seq_len = 20
        next_word = predict_next_word(model,tokenizer,text,max_seq_len)
        return render_template('predict.html',next_word=next_word)
    return render_template('predict.html',next_word='')  


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)