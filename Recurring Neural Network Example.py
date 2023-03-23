
############################################################################# 
###
### Import Required Packages 
###

import pandas as pd
from datetime import datetime 
import numpy as np
import random
import math


from nltk import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

import unidecode
from emoji import replace_emoji

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.text import Tokenizer
from keras import optimizers
from keras.callbacks import EarlyStopping 
from keras.models import load_model

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import set_config
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold




from matplotlib import pyplot
import matplotlib.pyplot as plt


#################################################################################################
###
###  Load in the required files
###

filepath = "C:\\Users\\matth\\OneDrive\\Desktop\\Independant Research Appendices\\"

nerload= pd.read_csv(filepath+"NER.csv")  
tweetsload= pd.read_csv(filepath+"tweets.csv")  

tweetshort = tweetsload[["Message","Sentiment"]]

##################################################################################################
###
### Create Functions used to Clean Data
###



############################
## Remove Accents
##

def removeacc(row):
    unaccented_string =unidecode.unidecode(row)
    return unaccented_string 


############################
## Replace words that are in NER list with respective category
## 
def wordfill(row):
    if row['NER']=="nan":
     return row['Name']
    else:  
     return row['NER']


############################
## Replace Emoji with Unicode
##

def unicode_escape(chars, data_dict):
    return chars.encode('unicode-escape').decode()+' '

###########################
## Pre Process each line of code
##

def token(line):
      #### Remove Accent
      line=replace_emoji(line, replace=unicode_escape)
      
      line=unidecode.unidecode(line)   
      
      #### Tokenize the list
      
      lists = regexp_tokenize(line, pattern=r"\s|[\.,;']", gaps=True)
      
      
      
      #### Apply NER replacement
      df = pd.DataFrame (lists, columns = ['Name'])
      joined = pd.merge(df, nerload, how='left', on = 'Name')
      joined['NER']=joined['NER'].astype(str)
      joined['Name']=joined.apply (lambda row: wordfill(row), axis=1)
      
      
      #### Convert to list
      lists=joined['Name'].tolist()   
      
      #### Apply word Stemming
      for i in range(0,len(lists)):
        
        ##### Need to avoid stemming NER replacements or Target
        if lists[i][0]=='|':
             lists[i]= lists[i]
        else:
          lists[i]=ps.stem(lists[i])
      
      #### Replace First and Second Name NER with |person| tag  
      detoken=TreebankWordDetokenizer().detokenize(lists)
      replaced=detoken.replace('|person-F| |person-S|','|person|')
      
      
      lists = regexp_tokenize(replaced, pattern=r"\s|[\.,;']", gaps=True)
      return lists




################
###
### Format NER list
###
nerload['Name']=nerload['Name'].apply (lambda row: removeacc(row))
nerload=nerload.rename(columns={"Unnamed: 1": "NER"})
nerload['Name']=nerload['Name'].str.lower()



################
###
###  Load Stop word list and remove negations to preserve Polartity Shift
### 

stop = stopwords.words('english')
stop.remove('no')
stop.remove('not')
stop.remove("isn't")
stop.remove("then")

#####################
###
### Load Word Stemmer
###

ps = PorterStemmer()

######################################
###
### Prepare full Transformation 
###



tweetshort = tweetsload
tweetshort['Message'] = tweetshort['Message'].str.replace('#','|hashtag| ')
textwork=tweetshort.iloc[59]['Message'].lower().replace('\n',' ') ##.strip() 
split=tweetshort.iloc[59]['Message'].split()
lowerstuff= [word.lower() for word in split]
stop = stopwords.words('english')
stop.remove('no')
stop.remove('not')
stop.remove("isn't")
stop.remove("then")
stopwordlest =[word for word in lowerstuff if word.lower() not in stop]

result = ' '.join(stopwordlest)


def transformtweet(row):
    #### Replace Hashtag
    row = row.replace('#','|hashtag| ')
    
    ####Split words into list
   
    splitrow=row.split()
    
    #### Lowercase the words
    lowerrow=[word.lower() for word in splitrow]
    
    #### Remove Stopwords 
    stopwordrow =[word for word in lowerrow if word.lower() not in stop]
    
    #### Join list back into formatted text
    row = ' '.join(stopwordrow)
    
    #### Replace unwanted characters
    row=row.replace('(?<=\w)[^\s\w\|](?![^\s\w])',' ')
    
    
    finalmessage=token(row)
    return finalmessage



#######################################################################
###
### Pre Processesing to Text 
###

tweetshort['PPMessage']=tweetshort['Message'].apply(transformtweet)



#######################################################################
###
### Neural network Design
###

sentences = tweetshort['PPMessage'].tolist()



#### Replace Negative Sentiment with 0 and Neutral and Postive with 1


tweetshort['Sentiment']=tweetshort['Sentiment'].replace('N',0).replace('Neutral',1).replace('P',1)
labels = tweetshort['Sentiment'].tolist()


########################################################
###
### Create Neural Network
###


max_words = 10000
embedding_dim = 16
max_length = 280
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"






#######################################
###
### Prepare Data
###

tokenizer = Tokenizer(num_words=max_words, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index


###########
### Tokenize text and leave only the 10,000 most popular words in sentences
###

sentence_sequences = tokenizer.texts_to_sequences(sentences)

##########
### Pad Text so each line is 280 characters long
###

sentence_padded=keras.preprocessing.sequence.pad_sequences(sentence_sequences, maxlen=max_length)

sentence_padded = np.array(sentence_padded)

labels = np.array(labels)



#rand=random.randint(0,2000)
rand=13
print("Random seed " + str(rand))
##########################################
#### Create Training and Testing data set

x_train, x_test, y_train, y_test  = train_test_split(sentence_padded, labels, test_size=0.1, random_state=rand)


##########################################
#### With remaining create a 10 fold cross validation model


ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=rand)
i=0
modelscores=pd.DataFrame(columns=[['ModelNumber','Precision','Recall','F1_Score','Accuracy','TotalRows','NegativeRows','Neg%']])
for train_index, val_index in ss.split(x_train, y_train):
    train_features, val_features = x_train[train_index], x_train[val_index]
    train_labels, val_labels = y_train[train_index], y_train[val_index]
    #### Save best version of model that best predicts the Valuation Accuracy
    
    
    model = Sequential()
    inputs = keras.Input(shape=(None,), dtype="int32")


    #### Word embedding

    model.add(layers.Embedding(max_words, 280, input_length=max_length))

    #### Bidirectional LSTM Layer with dropout
    model.add(layers.Bidirectional(layers.LSTM(30,dropout=0.3)))
 
    #### Dense Activation Layer with regularizer to reduce overfitting
    model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=4e-3, l2=4e-3),bias_regularizer=regularizers.l2(2e-3)))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()


    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(0.01),metrics=['accuracy'])


    checkpoint =  ModelCheckpoint("best_model"+str(i)+".hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1,save_weights_only=False)

    #### Early stop to avoid overfitting as Valuation accuracy decreases
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, mode='max') 
    callbacks_list = [checkpoint, early_stop]
    model.fit(train_features, train_labels, epochs=10,validation_data=(val_features, val_labels),callbacks=callbacks_list)
    
    loaded_model = load_model('best_model'+str(i)+'.hdf5')
    
    y_pred = (loaded_model.predict(val_features) > 0.5).astype("int32")
    
    nonneg=val_labels.sum()
    total=len(val_labels)
    
    modelscores.at[i,'ModelNumber']=i
    modelscores.at[i,'Precision']=precision_score(val_labels, y_pred)
    modelscores.at[i,'Recall']=recall_score(val_labels, y_pred)
    modelscores.at[i,'F1_Score']=f1_score(val_labels, y_pred)
    modelscores.at[i,'Accuracy']=accuracy_score(val_labels, y_pred)
    modelscores.at[i,'TotalRows']=total
    modelscores.at[i,'NegativeRows']=total-nonneg
    modelscores.at[i,'Neg%']=nonneg/total
    i=i+1



print(modelscores)

##################################################################################
###
###Testing 
###
y_test.sum()
modelscores2=pd.DataFrame(columns=[['ModelNumber','Precision','Recall','F1_Score','Accuracy','TotalRows','NegativeRows','Neg%']])

for i in range(0,10):
    
  loaded_model = load_model("best_model"+str(i)+".hdf5")
  y_predict = (loaded_model.predict(x_test) > 0.5).astype("int32")
  nonneg=y_test.sum()
  total=len(y_test)
  modelscores2.at[i,'ModelNumber']=i
  modelscores2.at[i,'Precision']=precision_score(y_test, y_predict)
  modelscores2.at[i,'Recall']=recall_score(y_test, y_predict)
  modelscores2.at[i,'F1_Score']=f1_score(y_test, y_predict)
  modelscores2.at[i,'Accuracy']=accuracy_score(y_test, y_predict)
  modelscores2.at[i,'TotalRows']=total
  modelscores2.at[i,'NegativeRows']=total-nonneg
  modelscores2.at[i,'Neg%']=nonneg/total
  



#######################
#####
##### Load best model for prediction

loaded_model = load_model('best_model4.hdf5')


y_pred = (loaded_model.predict(x_test) > 0.5).astype("int32")
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))





###########################################################################
####
#### Applying Model to Overall 
####

  
tweetsfull = pd.read_excel(filepath+"Extracted Tweets.xlsx", sheet_name='Sheet1')


#############
####
#### Try best model so far
####

tweetsfull['PPMessage']=tweetsfull['Message'].apply(transformtweet)

sentences = tweetsfull['PPMessage'].tolist()

#tweetsfull['Sentiment']=tweetsfull['Sentiment'].replace('N',0).replace('Neutral',1).replace('P',1)
#labels = tweetshort['Sentiment'].tolist()

tokenizer = Tokenizer(num_words=max_words, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index


###########
### Tokenize text and leave only the 10,000 most popular words in sentences
###

sentence_sequences = tokenizer.texts_to_sequences(sentences)

##########
### Pad Text so each line is 280 characters long
###

sentence_padded=keras.preprocessing.sequence.pad_sequences(sentence_sequences, maxlen=max_length)

sentence_padded = np.array(sentence_padded)


predictions = (loaded_model.predict(sentence_padded) > 0.5).astype("int32")

sentimentpd = pd.DataFrame(predictions,columns=['sentiment'])
full=tweetsfull.join(sentimentpd)

full.to_csv((filepath+"output.csv"  ),index = False)


