import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category = DeprecationWarning)



class Plot_fex:


    def __init__(self):
        self.combi = pd.read_csv('new_combi.csv')
        

    def h_extract(self, x):

        self.x = x
        hasht = []

        #loop words in tweet:
        for i in x:
            ht = re.findall(r"#[\w+]*", str(i))
            hasht.append(ht)

        return hasht

    def impl_hx(self):


        #extracting hashtags from non-offensive tweets
        ht_reg = self.h_extract(self.combi['clean_tweet'][self.combi['label'] == 0])

        #extracting hashtags from offensive tweets
        ht_neg = self.h_extract(self.combi['clean_tweet'][self.combi['label'] == 1])

        ht_reg = sum(ht_reg, [])       #unnesting
        ht_neg = sum(ht_neg, [])

        return ht_reg, ht_neg

    def plot(self, ht_reg, ht_neg):

        self.ht_reg = ht_reg
        self.ht_neg = ht_neg

        #plot ht_reg
        count_reg = nltk.FreqDist(ht_reg)
        df_reg = pd.DataFrame({'Hashtag': list(count_reg.keys()), 'Count': list(count_reg.values())})
        df_reg = df_reg.nlargest(columns = 'Count', n = 10)
        plt.figure(figsize = (16,5))
        ax = sns.barplot(data = df_reg, x = 'Hashtag', y = 'Count')
        ax.set(ylabel = 'Count')
        plt.show()

        #plot ht_neg
        count_neg = nltk.FreqDist(ht_neg)
        df_neg = pd.DataFrame({'Hashtag': list(count_neg.keys()), 'Count': list(count_neg.values())})
        df_neg = df_neg.nlargest(columns = 'Count', n = 10)
        plt.figure(figsize = (16,5))
        ax = sns.barplot(data = df_neg, x = 'Hashtag', y = 'Count')
        ax.set(ylabel = 'Count')
        plt.show()

        return

    def f_ex(self):
        self.bow_vectorizer = CountVectorizer(max_df = 0.90, min_df = 2, max_features = 1000, stop_words = 'english')
        self.bow = self.bow_vectorizer.fit_transform(self.combi['clean_tweet'].values.astype('U'))


        self.tfidf_vectorizer = TfidfVectorizer(max_df = 0.90, min_df = 2, max_features = 1000, stop_words = 'english')
        self.tfidf =self.tfidf_vectorizer.fit_transform(self.combi['clean_tweet'].values.astype('U'))


    @property
    def return_df(self):
        return self.combi
        

hx_plt_fex = Plot_fex()
#combi = hx_plt_fex.return_df()
ht_reg, ht_neg = hx_plt_fex.impl_hx()
hx_plt_fex.plot(ht_reg, ht_neg)
hx_plt_fex.f_ex()


from sklearn.model_selection import train_test_split

df = pd.read_csv('new_combi.csv')
df_train = pd.read_csv('train_E6oV3lV.csv')

p=Plot_fex()
p.f_ex()

train_data_bow = p.bow[:31962,:]
test_data_bow = p.bow[31962:,:]

train_data_tfidf = p.tfidf[:31962,:]
test_data_tfidf = p.tfidf[31962:,:]



#Z=p.bow
#X=train_data_tfidf
#Y= df_train.label


X_train = train_data_tfidf
X_test = test_data_tfidf
y_train = df_train["label"].values[:31962]
y_test = df["label"].values[31962:]

from sklearn import svm



def evaluate_on_test_data(model=None):
    predictions=model.predict(X_test)
    corr_classifications=0
    accuracy=model.score(X_train,y_train)
    return accuracy

model=svm.SVC(kernel="poly")
model.fit(X_train,y_train)
acc=evaluate_on_test_data(model)
print('Accuracy with kernel:','is:',acc*100)




        
        
        
        
        
        

        

        

        

        
        
