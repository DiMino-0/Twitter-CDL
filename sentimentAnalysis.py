# Import Libraries
from aiohttp import request
from textblob import TextBlob
import secrets
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
import requests
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

auth = tweepy.OAuth2BearerHandler(secrets.BearerToken)
api = tweepy.API(auth)
client = tweepy.Client(secrets.BearerToken)

# passedName = input("Enter the twitter user you wish to analyze: ")
# numberOfTweets = input("Enter the # of tweets you want to go back to 5-10: ")
passedName = "bts_bighit"
numberOfTweets = 10

positive = 0
negative = 0
neutral = 0
polarity = 0
tweetListFormatted = []
neutralList = []
negativeList = []
positiveList = []

user = client.get_user(username=passedName)
tweetListUnformatted = client.get_users_tweets(user.data.id, max_results = numberOfTweets)

def percentage(part,whole):
    return 100 * float(part)/float(whole)

for text in tweetListUnformatted.data:
    tweetListFormatted.append(str(text))
    analysis = TextBlob(str(text))
    score = SentimentIntensityAnalyzer().polarity_scores(str(text))
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negativeList.append(str(text))
        negative += 1
    elif pos > neg:
        positiveList.append(str(text))
        positive += 1
    elif pos == neg:
        neutralList.append(str(text))
        neutral += 1

positive = percentage(positive, numberOfTweets)
negative = percentage(negative, numberOfTweets)
neutral = percentage(neutral, numberOfTweets)
polarity = percentage(polarity, numberOfTweets)

positive = format(positive, ".1f")
negative = format(negative, ".1f")
neutral = format(neutral, ".1f")

#Number of Tweets (Total, Positive, Negative, Neutral)
tweetList = pd.DataFrame(tweetListFormatted)
positiveList = pd.DataFrame(positiveList)
negativeList = pd.DataFrame(negativeList)
neutralList = pd.DataFrame(neutralList)

#Pritning output to cmd
print("\nTweet List (",numberOfTweets,")\n\n\t", tweetListUnformatted.data, "\n\nParsing data...")

for text in tweetListUnformatted.data:
    print("\t", text, "\n") 

print("\nInfo about Tweets: ")
print("\ttotal number:" ,len(tweetList))
print("\tpositive number:" , len(positiveList))
print("\tnegative number: ", len(negativeList))
print("\tneutral number: ", len(neutralList))

#Creating PieCart
labels = ["Positive ["+str(positive)+"%]" , "Neutral ["+str(neutral)+"%]","Negative ["+str(negative)+"%]"]
sizes = [positive, neutral, negative]
colors = ["yellowgreen", "blue","red"]
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use("default")
plt.legend(labels)
plt.title("Sentiment Analysis Result For " + str(passedName))
plt.axis("equal")
my_circle=plt.Circle( (0,0), 0.7, color="white")
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

#Function to Create Wordcloud
def create_wordcloud(text):
    mask = np.array(Image.open("bulb.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color='white',
    mask = mask,
    max_words=3000,
    stopwords=stopwords,
    repeat=True)
    wc.generate(text)
    wc.to_file('wc.png')
    print("Word Cloud Saved Successfully")
    path="wc.png"
    sys.displayhook(Image.open(path))

create_wordcloud(positiveList.values)