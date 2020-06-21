
import re
import pandas as pd
import numpy as np
import nltk
import warnings
import string
from nltk.stem.porter import *

warnings.filterwarnings("ignore", category = DeprecationWarning)


class Preprocessing:

  train = pd.read_csv('train_E6oV3lV.csv')
  test = pd.read_csv('test_tweets_anuFYb8.csv')
  global combi
  combi = train.append(test)
  global input_txt
  global pattern


  def before(self, train):

        self.train = train
        print("data before pre-processing: ")
        print(train.head())
        for i in range(0,3): print(" ")

        return 


  def remove_pattern(self, input_txt, pattern):

        #input_txt = None
        #pattern = None
        r = re.findall(str(pattern),str(input_txt))
        for i in r:
            input_txt = re.sub(i,'',str(input_txt))

        return input_txt


  def pre_process1(self):
        
        #removing user handles:
        combi['clean_tweet'] = np.vectorize(self.remove_pattern)(combi['tweet'], "@[\w]*")
        
        #removing special characters, punctuatuons and numbers:
        combi['clean_tweet'] = combi['clean_tweet'].str.replace("[^a-zA-Z#]"," ")
        print("combi before len func:")
        print(" ")
        print(combi['clean_tweet'].head())
        for i in range(0,3): print(" ")
        
        #removing words length less than 3:
        combi['clean_tweet'] = combi['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        print("combi after len func:")
        print(" ")
        print(combi['clean_tweet'].head())
        for i in range(0,3): print(" ")
        combi.to_csv("new_combi.csv")

        return combi
        

  def pre_process2(self, combi):

        #tokenziing the tweets
        combi['clean_tweet'] = combi['clean_tweet'].apply(lambda x: x.split())
        print("tokenized tweet is: ")
        print(combi.head())
        for i in range(0,3): print(" ")

        stemmer = PorterStemmer()
        combi['clean_tweet'] = combi['clean_tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
        print("Tokenized data after stemming is: ")
        print(combi.head())
        for i in range(0,3): print(" ")

        return combi


process = Preprocessing()
process.before(Preprocessing.train)
combi = process.pre_process1()
tokenized_tweet = process.pre_process2(combi)
tokenized_tweet.to_csv("new_combi.csv")






