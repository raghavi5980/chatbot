# chatbot
# building a chatbot

import nltk
import numpy as np
import random
import string

import bs4 as bs
import urllib.request
import re

import pandas as pd
import nltk

print ("You have successfully imported pandas version "+pd.__version__)
print ("You have successfully imported nltk version "+nltk.__version__)

# load the csv file that contains the tweets on sexual harassments into a pandas dataframe
df_raw = pd.read_csv('./analysis-public-place-assaults-sexual-assaults-and-robberies-2015-csv.csv', encoding='latin-1')

print ("You have successfully loaded your csv file")

df_raw.head(5)

len(df_raw)

df_raw['Territorial_authority_area_2013_label'].head(8)

sample_tweet = df_raw.iloc[100]
print('Before tokenization:', sample_tweet)

sample_tweet = df_raw.iloc[100]
print('After tokenization:', sample_tweet)

stop = nltk.corpus.stopwords.words('english')
# Add a few more stop words we would like to remove here
stop.append('@')
stop.append('#')
stop.append('http')
stop.append(':')
stop

# Let us take two sentences to be the knowledge base, and one more which is a question.
Sentence_1='Workplace harassment is the belittling or threatening behavior directed at an individual worker or a group of workers'
Sentence_2='Physical harassment in the workplace takes many forms'
Question='In an informal survey among 50 employees in Singapore'

Vocabulary = ['workplace', 'is', 'among', 'informal', 'forms', 'individual', 'group']

bag_of_words = np.array([[1,1,1,1,0,0,0],
                        [0,1,1,0,1,1,0],
                        [0,1,1,1,0,0,1]])

bag_of_words[:3,:]

print (bag_of_words[-1,:].reshape(1, -1))

import matplotlib.pyplot as plt
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
plt.xlabel('workplace')
plt.ylabel('gender ratio')
plt.title('Histogram of sexual harassments')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Workplace_harassment')
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:
    article_text += para.text

article_text = article_text.lower()

f=open('workplace harassment.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower()# converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) #see previous section 1.2.5 lemmatization

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)

article_sentences = nltk.sent_tokenize(article_text)
article_words = nltk.word_tokenize(article_text)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_response(user_input):
    cookie_response = ''
    article_sentences.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(article_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        cookie_response = cookie_response + "I am sorry, I could not understand you"
        return cookie_response
    else:
        cookie_response = cookie_response + article_sentences[similar_sentence_number]
        return cookie_response

def greeting(sentence):
    for word in sentence.split(): # Looks at each word in your sentence
        if word.lower() in GREETING_INPUTS: # checks if the word matches a GREETING_INPUT
            return random.choice(GREETING_RESPONSES) # replies with a GREETING_RESPONSE

def response(user_response):
    
    cookie_response='' # initialize a variable to contain string
    sent_tokens.append(user_response) #add user response to sent_tokens
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') 
    tfidf = TfidfVec.fit_transform(sent_tokens) #get tfidf value
    vals = cosine_similarity(tfidf[-1], tfidf) #get cosine similarity value
    idx=vals.argsort()[0][-2] 
    flat = vals.flatten() 
    flat.sort() #sort in ascending order
    req_tfidf = flat[-2] 
    
    if(req_tfidf==0):
        cookie_response=cookie_response+"I am sorry! I don't understand you"
        return cookie_response
    else:
        cookie_response = cookie_response+sent_tokens[idx]
        return cookie_response

import datetime
def tell_time(sentence):
    for word in sentence.split():
        if word.lower() -- 'time':
            currentdt - datetime.datetime.now()
            return currentdt.strftime("%Y-%n-%d %H:%M:%S")

flag=True
print("cookie: My name is cookie. I will answer your queries about Workplace Harassment. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("cookie: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("cookie: "+greeting(user_response))
            else:
                print("cookie: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("cookie: Bye! take care..")

