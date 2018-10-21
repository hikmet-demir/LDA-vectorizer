import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pickle
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import copy
import nltk
import unicodedata
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

def py_asciify(text):
    return unicodedata.normalize('NFKD', text.replace(u'\u0131', 'i').replace(u"\u00A5",'y').replace(u"\u2122",' ').replace(u"\u00E2",'a').replace(u')', '\)').replace(u'(', '\(')).encode('ascii', 'ignore').decode('utf8').lower()

def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    docs= tokenizer.tokenize(docs)  # Split into words.
    #print(docs)
    docs = [token for token in docs if len(token) > 1]

    return docs

def trial_docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
        
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    return docs


stemmer = nltk.stem.SnowballStemmer('english')
def cleaner(text):
    text = py_asciify(str(text))
    tokenized = nltk.word_tokenize(text)
    regex = re.compile('[^a-z0-9!,.?\'"-:; ]')
    text = regex.sub('', text)
    if text == "nan" or text == "" or text == None:
        text = "empty"

    tokenized = nltk.word_tokenize(text)
    total = ""
    for i in tokenized:
        total+= stemmer.stem(i) + " "

    return total

df = pd.read_csv("data_lda_final.csv", header=[0], sep= '\t')

clear = df["tokenized"].tolist()
clear = [ str(i) for i in clear ]
clear = trial_docs_preprocessor(clear)
docs = clear

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)
print('Number of unique words in initital documents:', len(dictionary))

# Filter out words that occur less than 3 documents, or more than 70% of the documents.
dictionary.filter_extremes(no_below=3, no_above=0.70)
print('Number of unique words after removing rare and common words:', len(dictionary))

corpus = [dictionary.doc2bow(doc) for doc in docs]
# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token


num_topics=30
lda_model = LdaModel(corpus = corpus, id2word=id2word,alpha='auto', eta='auto', num_topics = num_topics)

pickle.dump(lda_model, open( "gensim_lda_model.p" , "wb" ))
pickle.dump(dictionary, open( "gensim_lda_dictionary.p" , "wb" ))
diccc = pickle.load( open( "gensim_lda_dictionary.p", "rb" ) )

s = ''

doc_topic = []
for i in range(len(corpus)):
    doc_topic.append(lda_model.get_document_topics(corpus[i]))
    
doc_topic_temp = []
for i in range(len(corpus)):
    temp = []
    for i in range(0,num_topics):
        a = 0
        temp.append(a)
    doc_topic_temp.append(temp)

longg = len(doc_topic)
for i in range(0,longg):
    #print(doc_topic[i])
    #print(i)
    kk = len(doc_topic[i])
    for a in range(kk):
        #print(i)
        doc_topic_temp[i][doc_topic[i][a][0]] = doc_topic[i][a][1]
    #print(doc_topic_temp[0])
    #break

lda_latents= []
for doc_l in doc_topic_temp:
    temp = ""
    for topic_l in range(len(doc_l)):
        temp += str(topic_l) + "|" + str(doc_l[topic_l]) + " "
    temp = temp[0:-1]
    lda_latents.append(temp)

df['lda_factors'] = lda_latents
df.to_csv("data_lda_final.csv", index=False, sep = '\t')