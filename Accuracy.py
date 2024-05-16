#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on the day we all start to love our self.

@author: Nikie Jo Deocampo
"""

import json
import pandas as pd
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

tweets_data = []
x = []
y = []
vectorizer = CountVectorizer(stop_words='english')

def retrieveTweet(data_url):
    tweets_data_path = data_url
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue
             
def retrieveProcessedData(Pdata_url):
    sent = pd.read_excel(Pdata_url)
    for i in range(len(tweets_data)):
        if tweets_data[i]['id'] == sent['id'][i]:
            x.append(tweets_data[i]['text'])
            y.append(sent['sentiment'][i])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')          

def nbTrain():
    from sklearn.naive_bayes import MultinomialNB
    start_timenb = time.time()
    train_features = vectorizer.fit_transform(x)
    actual = y
    nb = MultinomialNB()
    nb.fit(train_features, [int(r) for r in y])
    test_features = vectorizer.transform(x)
    predictions = nb.predict(test_features)
    nb_matrix = confusion_matrix(actual, predictions)
    print("\nNaive Bayes  Accuracy : \n", format(metrics.accuracy_score(actual, predictions) * 100), "%")
    print("Completion Speed", round((time.time() - start_timenb), 5))
    return nb_matrix

def datree():
    from sklearn import tree
    start_timedt = time.time()
    train_featurestree = vectorizer.fit_transform(x)
    actual1 = y
    test_features1 = vectorizer.transform(x)
    dtree = tree.DecisionTreeClassifier()
    dtree = dtree.fit(train_featurestree, [int(r) for r in y])
    prediction1 = dtree.predict(test_features1)
    dtree_matrix = confusion_matrix(actual1, prediction1)
    print("\nDecision tree Accuracy : \n", format(metrics.accuracy_score(actual1, prediction1) * 100), "%")
    print("Completion Speed", round((time.time() - start_timedt), 5))
    return dtree_matrix

def Tsvm():
    from sklearn.svm import SVC
    start_timesvm = time.time()
    train_featuressvm = vectorizer.fit_transform(x)
    actual2 = y
    test_features2 = vectorizer.transform(x)
    svc = SVC()
    svc = svc.fit(train_featuressvm, [int(r) for r in y])
    prediction2 = svc.predict(test_features2)
    svm_matrix = confusion_matrix(actual2, prediction2)
    print("\nSupport vector machine Accuracy : \n", format(metrics.accuracy_score(actual2, prediction2) * 100), "%")
    print("Completion Speed", round((time.time() - start_timesvm), 5))
    return svm_matrix

def knN():
    from sklearn.neighbors import KNeighborsClassifier
    start_timekn = time.time()
    train_featureskn = vectorizer.fit_transform(x)
    actual3 = y
    test_features3 = vectorizer.transform(x)
    kn = KNeighborsClassifier(n_neighbors=2)
    kn = kn.fit(train_featureskn, [int(i) for i in y])
    prediction3 = kn.predict(test_features3)
    knn_matrix = confusion_matrix(actual3, prediction3)
    print("\nK Nearest Neighbors Accuracy : \n", format(metrics.accuracy_score(actual3, prediction3) * 100), "%")
    print("Completion Speed", round((time.time() - start_timekn), 5))
    return knn_matrix

def RanFo():
    from sklearn.ensemble import RandomForestClassifier
    start_timerf = time.time()
    train_featuresrf = vectorizer.fit_transform(x)
    actual4 = y
    test_features4 = vectorizer.transform(x)
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf = rf.fit(train_featuresrf, [int(i) for i in y])
    prediction4 = rf.predict(test_features4)
    rf_matrix = confusion_matrix(actual4, prediction4)
    print("\nRandom Forest Accuracy : \n", format(metrics.accuracy_score(actual4, prediction4) * 100), "%")
    print("Completion Speed", round((time.time() - start_timerf), 5))
    return rf_matrix

def runall():     
    retrieveTweet('data/tweetdata.txt')  
    retrieveProcessedData('processed_data/output.xlsx')
    nb_matrix = nbTrain()
    dtree_matrix = datree()
    svm_matrix = Tsvm()
    knn_matrix = knN()
    rf_matrix = RanFo()

    # Display confusion matrices
    plt.figure(figsize=(15, 10))
    plt.subplot(231)
    plot_confusion_matrix(nb_matrix, classes=[-1, 0, 1], title='Confusion matrix for Naive Bayes classifier')
    plt.subplot(232)
    plot_confusion_matrix(dtree_matrix, classes=[-1, 0, 1], title='Confusion matrix for Decision Tree classifier')
    plt.subplot(233)
    plot_confusion_matrix(svm_matrix, classes=[-1, 0, 1], title='Confusion matrix for Support Vector Machine classifier')
    plt.subplot(234)
    plot_confusion_matrix(knn_matrix, classes=[-1, 0, 1], title='Confusion matrix for K Nearest Neighbors classifier')
    plt.subplot(235)
    plot_confusion_matrix(rf_matrix, classes=[-1, 0, 1], title='Confusion matrix for Random Forest classifier')
    plt.tight_layout()
    plt.show()

runall()
