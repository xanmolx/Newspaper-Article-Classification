import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import matplotlib.pyplot as plt
from subprocess import check_output

print("Taking data from Data set")

data = pd.read_csv('data/train.csv')
# print(data['group'])
data = data[['group','article']]
# train = data
train ,x = train_test_split(data,test_size = 0.95)

data = pd.read_csv('data/test.csv')
# print(data['group'])
data = data[['group','article']]
# test = data
test,x = train_test_split(data,test_size = 0.95)

print("Classifying data")
train_cl = []
for i in range(4):
    j=i+1
    train_cl.append(train[ train['group'] ==j])

for i in range(4):
    train_cl[i]=train_cl[i]['article']


test_cl = []
for i in range(4):
    j=i+1
    test_cl.append(test[ test['group'] ==j])


for i in range(4):
    test_cl[i]=test_cl[i]['article']

print("Removing Useless words")                
articles = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.article.split() if len(e) >= 3]
    words_without_stopwords = [word for word in words_filtered if not word in stopwords_set]
    articles.append((words_without_stopwords,row.group))


def get_words_in_articles(articles):
    all = []
    for (words, group) in articles:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

w_features = get_word_features(get_words_in_articles(articles))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features

print("Training")
training_set = nltk.classify.apply_features(extract_features,articles)
classifier = nltk.NaiveBayesClassifier.train(training_set)

# predict=nltk.NaiveBayesClassifier.classify(test[['article']])
test_set=nltk.classify.apply_features(extract_features,articles)
print("Testing")
nume,den=0,0
cnt = [0,0,0,0]
for i in range(4):
    for obj in test_cl[i]:
        res =  classifier.classify(extract_features(obj.split()))
        if(res == i+1): 
            cnt[i] = cnt[i] + 1
    # print (cnt[i],len(test_cl[i]))
    nume = nume + cnt[i]
    den = den + len(test_cl[i])
# print(nltk.classify.accuracy(classifier, test_set))
print ((nume/den)*100)
# print(np.mean(predict==test[['group']])
