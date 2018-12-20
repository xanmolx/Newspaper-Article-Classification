from flask import Flask, render_template, flash ,redirect, request, url_for, send_from_directory
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
from pathlib import Path
import numpy as np
import pandas as pd 
import pickle
import os
import shutil
from shutil import copyfile
from zipfile import ZipFile 
from flask_wtf import FlaskForm
from flask_ckeditor import CKEditorField
from wtforms import StringField, SubmitField
from keras.models import load_model
import tensorflow as tf
graph = tf.get_default_graph()


# from importlib import reload


app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'files/raw'
group_labels = ["raw","World","Sports","Business","Science"]


stop = stop_words.ENGLISH_STOP_WORDS
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

print("loading Bag of Words")
file = open('bow_knn_model.pkl', 'rb')
modelknn = pickle.load(file)#, encoding='latin1')
file.close()

file = open('bow_nb_model.pkl', 'rb')
modelnb = pickle.load(file)#, encoding='latin1')
file.close()

file = open('bow_svm_model.pkl', 'rb')
modelsvm = pickle.load(file)#, encoding='latin1')
file.close()

file = open('vectorizer.pkl', 'rb')
vectorizer = pickle.load(file)#, encoding='latin1')
file.close()
file_names = []

print("loading N grams")
file = open('ngrams_knn_model.pkl', 'rb')
modelknn_ng = pickle.load(file)#, encoding='latin1')
file.close()

file = open('ngrams_nb_model.pkl', 'rb')
modelnb_ng = pickle.load(file)#, encoding='latin1')
file.close()

file = open('ngrams_svm_model.pkl', 'rb')
modelsvm_ng = pickle.load(file)#, encoding='latin1')
file.close()

file = open('vectorizer_ngrams.pkl', 'rb')
vectorizer_ng = pickle.load(file)#, encoding='latin1')
file.close()
file_names = []

print("loading CNN")
cnn_model = load_model('cnn_model.h5')
file = open("cnn_tokenizer.pkl")
tokenizer = pickle.load(file)
file.close()

ng_cnn_model = load_model('ngrams_cnn_model.h5')
file = open("ngram_cnn_vectorizer.pkl")
ng_cnn_vectorizer = pickle.load(file)
file.close()


single_article_body = ""



class ArticleForm(Form):
    title = StringField('Title')#,[validators.Length(min=1,max=200)])
    body =  CKEditorField('') #,[validators.Length(min=30)])
    
@app.route('/singleArticle',methods=["POST","GET"])
def index():
    global single_article_body
    form = ArticleForm(request.form)
    if request.method == 'POST' and form.validate():
        title = form.title.data
        single_article_body = form.body.data
        return redirect(url_for('article'))

    return render_template('add_article.html', form=form)

def predict(text,model):
    test_clean_sentences = []
    line = text.strip()
    stop_free = " ".join([i for i in line.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    cleaned = processed.strip()
    test_clean_sentences.append(cleaned)
    if model == "bow_knn":
        Test = vectorizer.transform(test_clean_sentences)
        return group_labels[modelknn.predict(Test)[0]]
    if model == "bow_svm":
        Test = vectorizer.transform(test_clean_sentences)
        return group_labels[modelsvm.predict(Test)[0]]
    if model == "bow_nb":
        Test = vectorizer.transform(test_clean_sentences)
        return group_labels[modelnb.predict(Test)[0]]
    if model == "ng_knn":
        Test = vectorizer_ng.transform(test_clean_sentences)
        return group_labels[modelknn_ng.predict(Test)[0]]
    if model == "ng_svm":
        Test = vectorizer_ng.transform(test_clean_sentences)
        return group_labels[modelsvm_ng.predict(Test)[0]]
    if model == "ng_nb":
        Test = vectorizer_ng.transform(test_clean_sentences)
        return group_labels[modelnb_ng.predict(Test)[0]]
    if model == "cnn":
        Test =  tokenizer.texts_to_matrix(test_clean_sentences, mode='tfidf')
        tf.keras.backend.clear_session()
        # print("hello.........................")
        # print(Test.shape)
        # print("................................Bye")
        global graph
        with graph.as_default():
            return group_labels[np.argmax((cnn_model.predict(Test))[0]) + 1]
    if model == "ng_cnn":
        Test = ng_cnn_vectorizer.transform(test_clean_sentences)

        tf.keras.backend.clear_session()
        # print("hello.........................")
        # print(Test.shape)
        # print("................................Bye")
        global graph
        with graph.as_default():
            # print("ng cnn predict")
            print((ng_cnn_model.predict(Test)))
            return group_labels[np.argmax((ng_cnn_model.predict(Test))[0]) + 1]

@app.route('/article_catogerized')
def article():
    global single_article_body
    predictions = []
    models = ['bow_knn','bow_svm' , 'bow_nb','cnn','ng_knn','ng_svm' , 'ng_nb','ng_cnn']
    for model in models:
        print(predict(single_article_body,model))
        predictions.append(predict(single_article_body,model))
    # flash(x,'success')
    # predictions.append(-1)
    return render_template('article.html',preditions = predictions)
    # return redirect(url_for('about'))

@app.route('/')
def about():
    return render_template('about.html')

@app.route('/articles',methods=['POST','GET'])
def articles():
    if request.method == 'POST':
        global file_names
        file_names = []
        for f in request.files.getlist('files'):
            f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            file_names.append(f.filename)
        sel_mod  = request.form.get("sel_mod")
        print(sel_mod)
        return redirect(url_for('classifing',sel_mod=sel_mod))
    models_short = ['bow_knn','bow_svm' , 'bow_nb','cnn','ng_knn','ng_svm' , 'ng_nb','ng_cnn']
    models = ['Bag of Words K-NN','Bag of Words SVM' , 'Bag of Words NB','Bag of Words CNN','N-Grams KNN' , 'N-Grams SVM ','N-Grams NB','N-Grams CNN']
    return render_template('articles.html',models=models, models_short = models_short ,size = len(models_short) )
   
@app.route('/classifing')
def classifing():
    sel_mod = request.args['sel_mod']
    for file in file_names:
        # print(file)
        f = open('files/raw/'+file)
        x=predict(f.read(),sel_mod)
        print(x)
        copyfile('files/raw/'+file,'files/'+str(x)+"/"+file)
        f.close()
        # os.remove('files/raw/'+file)
        # print(f.read())
    return redirect(url_for('classified')) 

@app.route('/classified')
def classified():
    files = []
    for fol_name in group_labels[1:]:
        files.append(os.listdir('files/'+str(fol_name)))
    return render_template('dashboard.html',files=files)

@app.route('/classify/<string:filename>', methods=['GET','POST'])
def classify(filename):
    f = open('files/raw/'+filename)
    global single_article_body
    single_article_body  = f.read()
    f.close()
    # return "true"
    return redirect(url_for('article'))

@app.route('/delete',methods=['GET','POST'])
def delete():
    for fol_name in group_labels:
        fol = Path('files/'+str(fol_name))
        if fol.exists():
            shutil.rmtree('files/'+str(fol_name))
        os.makedirs('files/'+str(fol_name))
    return redirect(url_for('articles'))


def get_all_file_paths(directory): 
    file_paths = [] 
    for root, directories, files in os.walk(directory): 
        for filename in files: 
            filepath = os.path.join(root, filename) 
            file_paths.append(filepath) 
    return file_paths         
  
@app.route('/download',methods=['GET','POST'])
def download():
    #fol = Path('files/classified_files.zip')
    #if fol.exists():
     #   shutil.rmtree('files/classified_files.zip')
    directory = './files'
    file_paths = get_all_file_paths(directory) 
    with ZipFile('./files/classified_files.zip','w') as zip: 
        for file in file_paths: 
            zip.write(file) 
    return send_from_directory(directory="./files", filename="classified_files.zip")
    # return  ("./files", as_attachment=True)
if __name__ == '__main__':

    app.secret_key='secret123'
    app.run(debug=True)
    # app.run(host='0.0.0.0',port=5005)
