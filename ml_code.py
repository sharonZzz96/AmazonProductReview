#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 20:45:53 2018

@author: zhangyueying
"""

import pandas as pd
import numpy as np
import gzip
import math
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


'''
read json data
get dataframe
'''
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
              
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


'''
randomly select size=12000 samples with a least 20 votes
text: list of unprocessed text content
rating_low: list of 1 or 0 (if rating <= 2, then 1, else 0)
rating_high: list of 1 or 0 (if rating == 6, then 1, else 0)
Y: list of helpfulness: 
   for helpfulness < 0.4, Yi = - 1;
   for 0.4 <= helpfulness <= 0.6, Yi = 0
   for helpfulness > 0.6, Yi = 1
'''
def review_class(path, size):
    df = getDF(path)
    helpful = df['helpful'].tolist()
        # 1. list samples with at least 20 votes
    idxs = []
    for i in range(len(df.index)):
        if helpful[i][1] >= 20:
            idxs.append(i)

    review_class_all = df.loc[idxs]
    review_class = review_class_all.sample(n=size)
    text = review_class['reviewText'].tolist()
    rating_all = review_class['overall'].tolist()
    rating_low = [1 if rating <= 2 else 0 for rating in rating_all]
    rating_high = [1 if rating == 5 else 0 for rating in rating_all]
    Y_all = review_class['helpful'].tolist()
    Y = [1 if y[0]/y[1]>0.75 else -1 for y in Y_all]
    return text, rating_low, rating_high, Y


'''
Takes in a list of text ([[sample a], [sample b],....,]), then performs the following:
1. count text length (# of characters)
2. count # of sentences
3. calculate readability ARI = 4.71*(characters/words) + 0.5*(words/sentences) - 21.43
Return text length, num of sentences, readability score (all lists for all samples)
'''
def text_structure(texts):
    punc_sent = string.punctuation
    upper = string.ascii_uppercase
    text_length = []
    num_sentence = []
    readability = []
    for i in range(len(texts)):
        texti = texts[i]
        text_lengthi = 0
        num_sentencei = 0
        num_wordsi = len(texti.split(" "))
        
        for j in range(len(texti)):
            if texti[j].isalpha():
                text_lengthi += 1 
            if j <= len(texti) -3:
                if texti[j] in punc_sent and (texti[j+1] in upper or texti[j+2] in upper):
                    num_sentencei += 1
        if num_wordsi == 0:
            num_wordsi = 1
        if num_sentencei == 0 :
            num_sentencei = 1
            
        readabilityi = 4.71 * (text_lengthi/num_wordsi) + 0.5 * (num_wordsi/num_sentencei) - 21.43
        text_length.append(text_lengthi)
        num_sentence.append(num_sentencei)
        readability.append(readabilityi)
        
    return text_length, num_sentence, readability


'''
Takes in a list of text ([[sample a], [sample b],....,]), then performs the following:
1. Remove all punctuation
2. Remove all stopwords
3. Turn all words into lower case
4. Turn all words into its stem/lemmatize/unchanged state ??
5. Appear in at least A samples
Return the cleaned text as a list of words
'''  

# word_feature: list of unique processed words
# review_process: list of list of processed texts    
def text_process(text):  
    review_process = []
    review_string = []
    table = str.maketrans({key: None for key in string.punctuation})
    for sample in text:
        new_s = sample.translate(table)
        tokens_sample = word_tokenize(new_s)
        lmtzr_sample = WordNetLemmatizer()
       # stemmer_sample = PorterStemmer()
        sample_word = [lmtzr_sample.lemmatize(word.lower()) for word in tokens_sample if word.lower() not in stopwords.words('english')]
       # sample_word = [stemmer_sample.stem(word.lower()) for word in tokens_sample if word.lower() not in stopwords.words('english')] 
       # sample_word = [word.lower() for word in tokens_sample if word.lower() not in stopwords.words('english')]
        review_process.append(sample_word)
        review_string.extend(sample_word)
    word_feature = list(set(review_string))
    return word_feature, review_process


# vocabulary_list: list of unique processed words with appearance >= A
# all_words: dictionary of {unique processed words with appearance >= A: # of appearance}
# review_process: list of list of processed texts    
def word_count(text, A):
    word_feature, review_process = text_process(text)
    all_words = {}
    for sample in review_process:
        for w in list(set(sample)):
            all_words[w] = all_words.get(w,0) + 1
    vocabulary_list = []
    for item in all_words.items():
        if item[1] >= A:
            vocabulary_list.append(item[0])
    print('all_words', all_words)
    return vocabulary_list, all_words, review_process    
 
    
# content_all: list of list of tf_idf(for the words in vocabulary_list) for each text
def content_feature_extraction(voc_l, all_words, reivew_process):
    content_all = []
#==============================================================================
#     # bag of words (normalized with length of voc_l??)
#     for sample in review_process:
#         content = []
#         for w in voc_l:
#             content.append(sample.count(w)/len(voc_l))
#         content_all.append(content)
#==============================================================================
    # tf-idf
    for sample in review_process:
        frequency = []
        idf = []
        for w in voc_l:
            frequency.append(sample.count(w) / (len(sample) + 1)) # in case len(sample) = 0
            idf.append(math.log(len(review_process) / all_words[w]))
        max_fre = max(frequency)
        tf_np = np.array([0.5+0.5*i/(max_fre+0.001) for i in frequency])
        idf_np = np.array(idf)
        tf_idf = tf_np * idf_np
        content_all.append(tf_idf.tolist())
    return content_all
       
    
'''
takes in a list of unprocessed text
1. check if ! in text
2. check if ? in text
3. check if CAPITALIZED WORD in text
'''
def cap_exclamation(text):
    exclamation = [1 if '!' in sample else 0 for sample in text]
    question_mark = [1 if '?' in sample else 0 for sample in text]   
    cap = []
    for sample in text:
        has_cap = False
        try:
            sample_string = sample[0].split(' ')
            for word in sample_string:
                if word.isupper():
                    cap.append(1)
                    has_cap = True
                    break
            if has_cap == False:
                cap.append(0)
        except:
            cap.append(0)
    
    return exclamation, question_mark, cap
            

'''
model training:
     1. Naive Bayes
     2. SVM
     3. KNN
     4. Logistic Regression
'''
def logistic(X_train, Y_train, X_test):
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    preds = lr.predict(X_test)
    return preds

def GauNB(X_train, Y_train, X_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
#==============================================================================
#     nb = MultinomialNB()
#     nb.fit(X_train, Y_train)
#==============================================================================
    preds = gnb.predict(X_test)
    return preds
    
def S_V_M(X_train, Y_train, X_test, c):
    model = svm.SVC(C = c)
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    return preds

def knn(X_train, Y_train, X_test, k):
    neigh = KNeighborsClassifier(n_neighbors = k)
    neigh.fit(X_train, Y_train)
    preds = neigh.predict(X_test)
    return preds


'''
model performance:
    1. F-1 statistics
    2. validation error
'''
def test_result(Y_test, preds):
    return classification_report(Y_test, preds)

def validation_error(Y_test, preds):
    error = 0
    for i in range(len(preds)):
        if preds[i] != Y_test[i]:
            error += 1
    error_rate = error / len(preds)
    return error_rate   


 
sample_size = [3000, 5000, 8000, 10000, 12000]
validation_error_log = []
validation_error_gau = []
validation_error_svm = []
validation_error_knn = []

for size in sample_size:
    # get 12000 raw data
    text, rating_low, rating_high, Y = review_class('reviews_Apps_for_Android_5.json.gz', size)

    # generate feature vector X for all samples  e.g. X = [[sample a], [sample b],....]
    text_length, num_sentence, readability = text_structure(text)
    #voc_l, all_words, review_process = word_count(text, 20)
    #content_all = content_feature_extraction(voc_l, all_words, review_process)
    exclamation, question_mark, cap = cap_exclamation(text)
    X = []
    for i in range(len(Y)):
        X_sample = []
        X_sample.append(rating_low[i])
        X_sample.append(rating_high[i])
        X_sample.append(text_length[i])
        X_sample.append(num_sentence[i])
        X_sample.append(readability[i])
        X_sample.append(exclamation[i])
        X_sample.append(question_mark[i])
        X_sample.append(cap[i])
        #   X_sample.extend(content_all[i])
        X.append(X_sample)

    # choose 15% in 12000 as test set, 85% as train set    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=101)
#==============================================================================
#     ros = RandomOverSampler()
#     X_resampled, Y_resampled = ros.fit_sample(X_train, Y_train)
# #==============================================================================
# #     X_resampled, Y_resampled = SMOTE().fit_sample(X_train, Y_train)
# #==============================================================================
#     print(len(X_resampled), len(Y_resampled))
#==============================================================================
    X_resampled, Y_resampled = X_train, Y_train


# train and evaluate different models  
    preds = logistic(X_resampled, Y_resampled, X_test)
    test_report = test_result(Y_test, preds)
    val_error = validation_error(Y_test, preds)
    validation_error_log.append(val_error)
    if size == sample_size[-1]:
        print('logistics:\n', test_report)
        print('validation error rate:', val_error)
        print('')

    preds = GauNB(X_resampled, Y_resampled, X_test)
    test_report = test_result(Y_test, preds)
    val_error1 = validation_error(Y_test, preds)
    validation_error_gau.append(val_error1)
    if size == sample_size[-1]:
        print('gaussian naive bayes:\n', test_report)
        print('validation error rate:', val_error1)
        print('')

    preds2 = S_V_M(X_resampled, Y_resampled, X_test, 1)
    test_report2 = test_result(Y_test, preds2)
    val_error2 = validation_error(Y_test, preds2)
    validation_error_svm.append(val_error2)
    if size == sample_size[-1]:
        print('SVM:\n', test_report2)
        print('validation error rate:', val_error2)
        print('')

    preds3 = knn(X_resampled, Y_resampled, X_test, 5)
    test_report3 = test_result(Y_test, preds3)
    val_error3 = validation_error(Y_test, preds3)
    validation_error_knn.append(val_error3)
    if size == sample_size[-1]:
        print('KNN:\n', test_report3)
        print('validation error rate:', val_error3)
        print('')
    

plt.plot(sample_size,validation_error_log,label='Logistic Regression')
plt.plot(sample_size,validation_error_gau,label='Gaussian Naive Bayes')
plt.plot(sample_size,validation_error_svm,label='SVM')
plt.plot(sample_size,validation_error_knn,label='KNN')
plt.legend()
plt.xlabel('data size')
plt.ylabel('validation error')
plt.show() 
