#!/usr/bin/env python
# coding: utf-8

# In[29]:


# import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re


# In[30]:


def tokenize(sentence):
    a = word_tokenize(sentence)
    a = ' '.join(a)
    a = re.sub('[^a-zA-Z\s+\d+]', '', sentence)
    return a


# In[31]:


df = pd.read_csv('/Users/aamod/Aamod/UMass/Fall 2020/CS 490A/Project/data.csv', index_col=[0])
df = df.drop('id', axis = 1)
df = df.dropna()
df['tok_title'] = df['title'].apply(tokenize)
df['tok_comments'] = df['comments'].apply(tokenize)
df.isLeft.value_counts()


# In[32]:


# df['combined'] = df[['Subreddit', 'title', 'comments']].apply(lambda x: ' '.join(x), axis=1)
df['combined'] = df[['title', 'comments']].apply(lambda x: ' '.join(x), axis=1)
df.to_csv('tokenized_data.csv')
df.shape


# In[33]:


def predict(col_name, test_split):
    x = df[col_name]
    y = df['isLeft']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_split)#, random_state = 42)

    count_vector = CountVectorizer(stop_words = 'english')

    x_train = count_vector.fit_transform(x_train)
    x_test = count_vector.transform(x_test)
    
    model = MultinomialNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy  = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    recall = recall_score(y_test, pred)
    return pred, accuracy, precision, f1, recall


# In[34]:


def run_mult(times, column, test_split):
    accuracy = 0
    precision = 0
    f1 = 0
    recall = 0
    for i in range(times):
        a,b,c,d,e= predict(column, test_split)
        accuracy += b
        precision += c
        f1 += d
        recall += e

    accuracy /= times
    precision /= times
    f1 /= times
    recall /= times
    
    return accuracy, precision, f1, recall
        


# In[35]:


def get_avg_values(num_runs, test_split, to_print = False):
    ta, tp, tf, tr = run_mult(num_runs, 'tok_title', test_split)
    ca, cp, cf, cr = run_mult(num_runs, 'tok_comments', test_split)
    sa, sp, sf, sr = run_mult(num_runs, 'Subreddit', test_split)
    coa, cop, cof, cor = run_mult(num_runs, 'combined', test_split)
    
    if to_print:
        print('Accuracy title: ', format(ta.round(3)))
        print('Precision title: ', format(tp.round(3)))
        print('f1 title: ', format(tf.round(3)))
        print('recall title: ', format(tr.round(3)))
        print()
        print('Accuracy comments: ', format(ca.round(3)))
        print('Precision comments: ', format(cp.round(3)))
        print('f1 comments: ', format(cf.round(3)))
        print('recall comments: ', format(cr.round(3)))
        print()
        print('Accuracy subreddit: ', format(sa.round(3)))
        print('Precision subreddit: ', format(sp.round(3)))
        print('f1 subreddit: ', format(sf.round(3)))
        print('recall subreddit: ', format(sr.round(3)))
        print()
        print('Accuracy combined: ', format(coa.round(3)))
        print('Precision combined: ', format(cop.round(3)))
        print('f1 combined: ', format(cof.round(3)))
        print('recall combined: ', format(cor.round(3)))
    
    return [[ta,tp,tf,tr], [ca,cp,cf,cr], [sa,sp,sf,sr], [coa,cop,cof,cor]]


# In[44]:


metrics = get_avg_values(50, 0.2, True)


# In[37]:


## FIGURE OUT BEST SPLIT
def get_split():
    test_vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    accuracy = {'title':[],'comments':[],'subreddit':[],'combined':[]}
    precision = {'title':[],'comments':[],'subreddit':[],'combined':[]}
    f1 = {'title':[],'comments':[],'subreddit':[],'combined':[]}
    recall = {'title':[],'comments':[],'subreddit':[],'combined':[]}

    for test_split in test_vals:
        metrics = get_avg_values(1, test_split)

        accuracy['title'].append(metrics[0][0])
        precision['title'].append(metrics[0][1])
        f1['title'].append(metrics[0][2])
        recall['title'].append(metrics[0][3])

        accuracy['comments'].append(metrics[1][0])
        precision['comments'].append(metrics[1][1])
        f1['comments'].append(metrics[1][2])
        recall['comments'].append(metrics[1][3])

        accuracy['subreddit'].append(metrics[2][0])
        precision['subreddit'].append(metrics[2][1])
        f1['subreddit'].append(metrics[2][2])
        recall['subreddit'].append(metrics[2][3])

        accuracy['combined'].append(metrics[3][0])
        precision['combined'].append(metrics[3][1])
        f1['combined'].append(metrics[3][2])
        recall['combined'].append(metrics[3][3])
    
    return accuracy, precision, f1, recall


# In[38]:


accuracy, precision, f1, recall = get_split()
test_vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


# In[39]:


import matplotlib.pyplot as plt

a_avg = [(a + b + c)/3 for a,b,c in zip(accuracy['title'], accuracy['comments'], accuracy['combined'])]
a = 0.35

plt.plot(test_vals, accuracy['title'], label = "Title", alpha = a)
plt.plot(test_vals, accuracy['comments'], label = "Comments", alpha = a)
plt.plot(test_vals, accuracy['combined'], label = "Combined", alpha = a)
plt.plot(test_vals, a_avg, label = 'Average')

plt.title('Accuracy v test/train split')
plt.legend()
plt.xlabel('Test split')
plt.ylabel('Accuracy')


# In[40]:


p_avg = [(a + b + c)/3 for a,b,c in zip(precision['title'], precision['comments'], precision['combined'])]
a = 0.35

plt.plot(test_vals, precision['title'], label = "Title", alpha = a)
plt.plot(test_vals, precision['comments'], label = "Comments", alpha = a)
plt.plot(test_vals, precision['combined'], label = "Combined", alpha = a)
plt.plot(test_vals, p_avg, label = 'Average')

plt.title('precision v test/train split')
plt.legend()
plt.xlabel('Test split')
plt.ylabel('precision')
plt.show()


# In[41]:


f_avg = [(a + b + c)/3 for a,b,c in zip(f1['title'], f1['comments'], f1['combined'])]
a = 0.35

plt.plot(test_vals, f1['title'], label = "Title", alpha = a)
plt.plot(test_vals, f1['comments'], label = "Comments", alpha = a)
plt.plot(test_vals, f1['combined'], label = "Combined", alpha = a)
plt.plot(test_vals, f_avg, label = 'Average')

plt.title('f1 v test/train split')
plt.legend()
plt.xlabel('Test split')
plt.ylabel('f1')
plt.show()


# In[42]:


r_avg = [(a + b + c)/3 for a,b,c in zip(recall['title'], recall['comments'], recall['combined'])]
a = 0.35
plt.plot(test_vals, recall['title'], label = "Title", alpha = a)
plt.plot(test_vals, recall['comments'], label = "Comments", alpha = a)
plt.plot(test_vals, recall['combined'], label = "Combined", alpha = a)
plt.plot(test_vals, r_avg, label = 'Average')

plt.title('recall v test/train split')
plt.legend()
plt.xlabel('Test split')
plt.ylabel('recall')

plt.show()


# In[45]:


metrics_avg = [(a+b+c+d)/4 for a,b,c,d in zip(a_avg, f_avg, p_avg, p_avg)]
a = 0.3
plt.plot(test_vals, a_avg, label = 'Accuracy', alpha=a)
plt.plot(test_vals, f_avg, label = 'f1 Score', alpha=a)
plt.plot(test_vals, metrics_avg, label = 'Average', color = 'black')
plt.plot(test_vals, p_avg, label = 'Precision', alpha=a)
plt.plot(test_vals, p_avg, label = 'Recall', alpha=a)

plt.legend()
plt.xlabel('Test split')
plt.ylabel('Percentage')
plt.show()

