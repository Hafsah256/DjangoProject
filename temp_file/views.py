from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from .models import DB_model
from .models import ItemSelector, DataFrameToArrayTransformer
from sklearn.base import BaseEstimator, TransformerMixin
#from .model_training import evaluation_summary
#function
from urlextract import URLExtract
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import uuid
import re
from textblob import TextBlob
from tweepy import Stream
import os
from tweepy import API
from tweepy import Cursor
import tweepy as tw
import spacy
import en_core_web_sm
import pandas as pd
from twython import Twython
import json
import numpy as np
import csv
from sklearn.externals import joblib
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from django.contrib.auth.models import User,auth
import seaborn as sns
import string
import nltk
import warnings
from nltk.stem import PorterStemmer
#%matplotlib inline
#warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.corpus import stopwords
from collections import Counter
import os



temp=[]
credentials = {}
credentials['CONSUMER_KEY'] = "YoknJGgd0DYnUe4m7Z4gSDTK3"
credentials['CONSUMER_SECRET'] = "RU83Rq7hg2aKreuzZzAdzvR9QFCQfk0YMauJPRwI5D8GvgOg4g"
credentials['ACCESS_TOKEN'] = "1275391205598277633-dZD2nyYAqQG3Pb1avTIBvhfsQy0Q6G"
credentials['ACCESS_SECRET'] = "JQz2x78eetOPrFYYh8guwuLtsbtADD2f9zcnxYKddNiNo"
with open("twitter_credentials.json", "w") as file:
    json.dump(credentials, file)
    auth = tw.OAuthHandler(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])
    auth.set_access_token(credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])
    api = tw.API(auth, wait_on_rate_limit=True)
python_tweets = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])

list_of_objects=[]
queryrequest=[]
words=[]
stop = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")

from sklearn.base import BaseEstimator, TransformerMixin
# Create your models here
class DataFrameToArrayTransformer(BaseEstimator,TransformerMixin):
    pass
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.    """
    pass

#if __name__ == "__main__":
#model = joblib.load('bernoullinb.joblib')
#print('ajhd',model)
 #   print(model.predict("this is not working @dead"))

def home(request):
    obj = DB_model.objects.all()
    print("iterating over obj")
    for n in obj:
        if n.topics not in words:
            words.append(n.topics)

    print('words = ',words)
    return render(request,"homepage.html",{'list_':words})#,{'list':list_thing})#, {{'data_list':list_thing}})#HttpResponse("Hell world!!!!!")
def back(request):
    obj = DB_model.objects.all()
    print("iterating over obj")
    for n in obj:
        if n.topics not in words:
            words.append(n.topics)
    queryrequest.clear()
    return render(request, "homepage.html", {'list_': words})
def radiobtn(request):
    for i in list_of_objects:
        if i.topics in queryrequest:
            i.predictedlabel = request.POST.get(str(i.userID))
    queryrequest.clear()
    for n in obj:
        if n.topics not in words:
            words.append(n.topics)
    return render(request,"temp.html",{'list_':list_of_objects})#, {'col':"jhsdjkhfakdfa;ofosdfa"})
def search (request):
    #print(model)
    query = request.POST['search']
    queryrequest.append(query)
    print('query req ',queryrequest)
    dataframe = fetching(query)
    dataframe=AddFeatures(dataframe)
    length1 = len(dataframe)
    print(dataframe)

    for index, row in dataframe.iterrows():
        obj_index = DB_model()
        obj_index.userID =uuid.uuid4()
        obj_index.userName = (row['userName'])
        obj_index.text=(row['text'])
        obj_index.textLen=(row['textLen'])
        obj_index.retweetsCount=(row['retweetsCount'])
        obj_index.favoriteCount=(row['favoriteCount'])
        obj_index.source=(row['source'])
        obj_index.language=(row['language'])
        obj_index.favourited = (row['favourited'])
        obj_index.retweeted=(row['retweeted'])
        obj_index.userLocation= (row['userLocation'])
        temp = (row['URL'])
        if not temp:
            obj_index.URL = 'https://twitter.com/'
        else:
            obj_index.URL = temp
        obj_index.userfollowers_count=(row['userfollowers_count'])
        obj_index.userfriends_count=(row['userfriends_count'])
        obj_index.userListed_count=(row['userListed_count'])
        obj_index.userFavorites_count=(row['userFavorites_count'])
        obj_index.userStatuses_count=(row['userStatuses_count'])
        obj_index.userVerified=(row['userVerified'])
        obj_index.userProtected=(row['userProtected'])#userProtected[i]
        obj_index.sentiment = (row['sentiment'])#sentiment[i]
        obj_index.predictedlabel=-2
        obj_index.userHomelink = 'https://twitter.com/'+(row['screenName'])
        obj_index.user_profileImg = (row['imgUrl'])
        obj_index.topics= query
        obj_index.save()
        list_of_objects.append(obj_index)
    temp=[]
    for i in list_of_objects:
        if i.topics == query:
            temp.append(i)
    context ={'que':temp}
    return render(request, "res.html", context)
#used to add the additional features we have in our pipeline
def AddFeatures(df):
    analyzer = SentimentIntensityAnalyzer()
    list_ = []
    for sentence in df['text']:
        vs = analyzer.polarity_scores(sentence)
        list_.append(vs)
    neg = pos = neu = compound = []
    for i in list_:
        neg.append(i['neg'])
        pos.append(i['pos'])
        neu.append(i['neu'])
        compound.append(i['compound'])
    df['negative'] = pd.DataFrame(neg, columns=['negative'])
    df['positive'] = pd.DataFrame(pos, columns=['positive'])
    df['compound'] = pd.DataFrame(compound, columns=['compound'])
    df['neutral'] = pd.DataFrame(neu, columns=['neutral'])
    df['spl'] = df['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
    df['processedtext'] = df['text'].str.replace('[^\w\s]', '')
    df['processedtext'] = df['processedtext'].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop))
    stemmer = PorterStemmer()
    df['processedtext'] = df['processedtext'].apply(
        lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    temp = []
    entities = []
    numOfEntities = []
    for i in df['processedtext']:
        temp.append(nlp(i))
    for i in temp:
        sent = ''
        counter = 0
        for word in i.ents:
            counter = counter + 1
            sent = sent + " " + word.label_
            # print(word.text,word.label_)
        entities.append(sent)
        numOfEntities.append(counter)

    df['entities'] = pd.DataFrame(entities, columns=['entities'])
    df['numOfEntities'] = pd.DataFrame(numOfEntities, columns=['numOfEntities'])

    df.replace(r'^\s*$', "none", regex=True)
    return df
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
def analyze_sentiment(tweet):
    ana = TextBlob(clean_tweet(tweet))
    # ana = TextBlob()
    if ana.sentiment.polarity > 0:
        return 1
    elif ana.sentiment.polarity == 0:
        return 0
    else:
        return -1
def fetching(search_words, numberofitem=10):
    extractor = URLExtract()
    date_since = "2019-12-16"
    new_search = search_words + " -filter:retweets"
    print(new_search)
    tweets = tw.Cursor(api.search, q=new_search, lang="en", since=date_since).items(numberofitem)
    id = []
    textLen = []
    retweetsCount = []
    favoriteCount = []
    source = []
    language = []
    text = []
    retweeted = []
    favourited = []
    date = []
    name = []
    screenName = []
    location = []
    url = []
    followers_count = []
    friends_count = []
    listed_count = []
    favorite_count = []
    statuses_count = []
    verified = []
    prot = []
    senti = []
    imgurl=[]
    raw_tweet = []
    for t in tweets:
        raw_tweet.append(t)
        id.append(t.id)
        text.append(t.text)
        textLen.append(len(t.text))
        retweetsCount.append(t.retweet_count)
        favoriteCount.append(t.favorite_count)
        source.append(t.source)
        language.append(t.lang)
        date.append(t.created_at)
        favourited.append(t.favorited)
        retweeted.append(t.retweeted)
        name.append(t.user.name)
        imgurl.append(t.user.profile_image_url)
        screenName.append(t.user.screen_name)
        location.append(t.user.location)
        if t.user.url:  # not t.user.url:
            temp = ""
            for url_ in extractor.gen_urls(t.text):
                temp = url_
            if temp:
                url.append(url_)
            else:
                url.append(t.user.url)
        else:
            temp = ""
            for url_ in extractor.gen_urls(t.text):
                temp = url_
            if temp:
                url.append(temp)
            else:
                url.append('https://twitter.com/')

        followers_count.append(t.user.followers_count)
        friends_count.append(t.user.friends_count)
        listed_count.append(t.user.listed_count)
        favorite_count.append(t.user.favourites_count)
        statuses_count.append(t.user.statuses_count)
        prot.append(t.user.protected)
        verified.append(t.user.verified)
        senti.append(analyze_sentiment(t.text))

    df = pd.DataFrame(name, columns=['userName'])
    df['userID'] = pd.DataFrame(id, columns=['userID'])
    df['text'] = pd.DataFrame(text, columns=['text'])
    df['textLen'] = pd.DataFrame(textLen, columns=['textLen'])
    df['retweetsCount'] = pd.DataFrame(retweetsCount, columns=['retweetsCount'])
    df['favoriteCount'] = pd.DataFrame(favoriteCount, columns=['favoriteCount'])
    df['source'] = pd.DataFrame(source, columns=['source'])
    df['language'] = pd.DataFrame(language, columns=['language'])
    df['date'] = pd.DataFrame(date, columns=['date'])
    df['favourited'] = pd.DataFrame(favourited, columns=['favourited'])
    df['retweeted'] = pd.DataFrame(retweeted, columns=['retweeted'])
    df['userLocation'] = pd.DataFrame(location, columns=['userLocation'])
    df['URL'] = pd.DataFrame(url, columns=['URL'])
    df['userfollowers_count'] = pd.DataFrame(followers_count, columns=['userfollowers_count'])
    df['userfriends_count'] = pd.DataFrame(friends_count, columns=['userfriends_count'])
    df['userListed_count'] = pd.DataFrame(listed_count, columns=['userListed_count'])
    df['userFavorites_count'] = pd.DataFrame(favorite_count, columns=['userFavorites_count'])
    df['userStatuses_count'] = pd.DataFrame(statuses_count, columns=['userStatuses_count'])
    df['userVerified'] = pd.DataFrame(verified, columns=['userVerified'])
    df['userProtected'] = pd.DataFrame(prot, columns=['userProtected'])
    df['sentiment'] = pd.DataFrame(senti, columns=['sentiment'])
    df['rawTweet'] = pd.DataFrame(raw_tweet, columns=['rawTweet'])
    df['screenName'] = pd.DataFrame(screenName, columns=['screenName'])
    df['imgUrl'] = pd.DataFrame(imgurl, columns=['imgUrl'])
    return df


def retraining_model(length_of_data):
    print(data_frame.head(3))
    db_obj = DB_model.objects.all()
    print("in retrain len of db_obj")
    print(len(db_obj))
    id1=[]
    text1=[]
    textLen1 = []
    retweetsCount1 = []
    favoriteCount1 = []
    source1 = []
    language1 = []
    retweeted1 = []
    favourited1 = []
    date1 = []
    name1 = []
    location1 = []
    url1 = []
    followers_count1 = []
    friends_count1 = []
    listed_count1 = []
    favorite_count1 = []
    statuses_count1 = []
    verified1 = []
    prot1 = []
    senti1 = []
    label1 = []
    number1=[]
    counter = length_of_data
    for i in db_obj:
        number1.append(counter)
        counter=counter+1
        id1.append(float(i.twitterID))
        name1.append(i.userName)
        text1.append(i.text)
        textLen1.append(int(i.textLen))
        print('textlen type',type(textLen1))
        if i.retweeted == 'False':
            retweeted1.append(0)
        else:
            retweeted1.append(1)
        print('retweeted value ',i.retweeted,'retweeted type', type(retweeted1))
        if i.favourited == 'False':
            favourited1.append(0)
        else:
            favourited1.append(1)
        #favourited.append(int(i.favourited))
        print('favourited value ',i.favourited,'favourited type', type(favourited1))
        source1.append(i.source)
        language1.append(i.language)
        date1.append(i.date)
        location1.append(i.userLocation)
        retweetsCount1.append(int(i.retweetsCount))
        print('retweetsCount value ', i.retweetsCount, 'favourited type', type(retweetsCount1))
        favoriteCount1.append(int(i.favoriteCount))
        print('favoriteCount value ', i.favoriteCount, 'favourited type', type(favoriteCount1))
        url1.append(i.URL)
        followers_count1.append(int(i.userfollowers_count))
        print('userfollowers_count value ', i.userfollowers_count, 'favourited type', type(followers_count1))
        friends_count1.append(int(i.userfriends_count))
        print('userfriends_count value ', i.userfriends_count, 'favourited type', type(friends_count1))
        listed_count1.append(int(i.userListed_count))
        print('userListed_count value ', i.userListed_count, 'favourited type', type(listed_count1))
        favorite_count1.append(int(i.userFavorites_count))
        print('userFavorites_count value ', i.userFavorites_count, 'favourited type', type(favorite_count1))
        statuses_count1.append(int(i.userStatuses_count))
        print('statuses_count1 value ', i.userStatuses_count, 'favourited type', type(statuses_count1))
        if i.userVerified == 'False':
            verified1.append(0)
        else:
            verified1.append(1)
        print('userVerified value ', i.userVerified, 'favourited type', type(verified1))
        if i.userProtected == 'False':
            prot1.append(0)
        else:
            prot1.append(1)
        print('userProtected value ', i.userProtected, 'favourited type', type(prot1))

        #verified.append(int(i.userVerified))
        #prot.append((i.userProtected))
        senti1.append(int(i.sentiment))
        label1.append(int(i.predictedlabel))
        print('senti1 value ', i.sentiment, 'favourited type', type(senti1))
        print('label1 value ', i.predictedlabel, 'favourited type', type(label1))

    df = pd.DataFrame(name1, columns=['userName'])
    #df['userName'] = pd.DataFrame(name, columns=['userName'])
    df['userID'] = pd.DataFrame(id1, columns=['userID'])
    df['text'] = pd.DataFrame(text1, columns=['text'])
    df['textLen'] = pd.DataFrame(textLen1, columns=['textLen'])
    df['retweetsCount'] = pd.DataFrame(retweetsCount1, columns=['retweetsCount'])
    df['favoriteCount'] = pd.DataFrame(favoriteCount1, columns=['favoriteCount'])
    df['source'] = pd.DataFrame(source1, columns=['source'])
    df['language'] = pd.DataFrame(language1, columns=['language'])
    df['date'] = pd.DataFrame(date1, columns=['date'])
    df['favourited'] = pd.DataFrame(favourited1, columns=['favourited'])
    df['retweeted'] = pd.DataFrame(retweeted1, columns=['retweeted'])
    df['userLocation'] = pd.DataFrame(location1, columns=['userLocation'])
    df['URL'] = pd.DataFrame(url1, columns=['URL'])
    df['userfollowers_count'] = pd.DataFrame(followers_count1, columns=['userfollowers_count'])
    df['userfriends_count'] = pd.DataFrame(friends_count1, columns=['userfriends_count'])
    df['userListed_count'] = pd.DataFrame(listed_count1, columns=['userListed_count'])
    df['userFavorites_count'] = pd.DataFrame(favorite_count1, columns=['userFavorites_count'])
    df['userStatuses_count'] = pd.DataFrame(statuses_count1, columns=['userStatuses_count'])
    df['userVerified'] = pd.DataFrame(verified1, columns=['userVerified'])
    df['userProtected'] = pd.DataFrame(prot1, columns=['userProtected'])
    df['sentiment'] = pd.DataFrame(senti1, columns=['sentiment'])
    df['label'] = pd.DataFrame(label1, columns=['label'])

    existing_data_frame = pd.read_csv('data.csv')
    print('existing dataframe datatypes', existing_data_frame.dtypes)
    #frames =[existing_data_frame,df]
    #new_data_frame = pd.concat(frames)
    new_data_frame = existing_data_frame.append(df)


    print('----------------------------------\n')
    #print(new_data_frame)
    #print(len(new_data_frame))
    print('new dataframe datatypes', new_data_frame.dtypes)
    labels = (new_data_frame['label'])
    # nan values are replaced by this
    new_data_frame["source"].fillna("Twitter Web App", inplace=True)
    new_data_frame["userLocation"].fillna("No Location", inplace=True)
    new_data_frame["URL"].fillna("https://twitter.com/home", inplace=True)

    labels.dropna(inplace=True)
    new_data_frame.drop(columns='Unnamed: 0', inplace=True)
    new_data_frame.drop(columns='label', inplace=True)
    print('--------------after dropping columns--------------------\n')
    print(new_data_frame)
    print(labels)
    print(len(new_data_frame))
    new_data_frame = AddFeatures(new_data_frame)
    print("new features added")

    X_train, X_test, y_train, y_test = train_test_split(
        new_data_frame, labels, test_size=0.3, random_state=0)
    print("printing x_train",X_train)

    pipeline_feature_74 = Pipeline([
        ('union', FeatureUnion(
            [
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('one-hot',
                     CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
                ])),

                ('entities', Pipeline([
                    ('selector', ItemSelector(key='entities')),
                    ('one-hot',
                     CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
                ])),

                ('URL', Pipeline([
                    ('selector', ItemSelector(key='URL')),
                    ('one-hot',
                     CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
                ])),
                ('userfollowers_count', Pipeline([
                    ('selector', ItemSelector(key='userfollowers_count')),
                    ('array', DataFrameToArrayTransformer()),
                    # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
                ])),
                ('favoriteCount', Pipeline([
                    ('selector', ItemSelector(key='favoriteCount')),
                    ('array', DataFrameToArrayTransformer()),
                ])),
                ('sentiment', Pipeline([
                    ('selector', ItemSelector(key='sentiment')),
                    ('array', DataFrameToArrayTransformer()),
                ])),
                ('source', Pipeline([
                    ('selector', ItemSelector(key='source')),
                    ('one-hot',
                     CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
                ])),
                ('neg', Pipeline([
                    ('selector', ItemSelector(key='negative')),
                    ('array', DataFrameToArrayTransformer()),
                    # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
                ])),
                ('pos', Pipeline([
                    ('selector', ItemSelector(key='positive')),
                    ('array', DataFrameToArrayTransformer()),
                    # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
                ])),

                ('textlen', Pipeline([
                    ('selector', ItemSelector(key='textLen')),
                    ('array', DataFrameToArrayTransformer()),
                    # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
                ])),
                ('processedtext', Pipeline([
                    ('selector', ItemSelector(key='processedtext')),
                    ('one-hot',
                     CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
                ])),

                ('spl', Pipeline([
                    ('selector', ItemSelector(key='spl')),
                    ('array', DataFrameToArrayTransformer()),
                ])),

                ('userLocation', Pipeline([
                    ('selector', ItemSelector(key='userLocation')),
                    ('one-hot',
                     CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
                ]))
            ])

         ),
        ('rTree', RandomForestClassifier(random_state=10))
    ])
    pipeline_feature_74.fit(X_train, y_train)
    result074 = pipeline_feature_74.predict(X_test)
    print(evaluation_summary("Results with new dataset are ",result074,y_test))
    #df.to_csv('data.csv')
    #print('after to_csv')
    #print(df)

    joblib.dump(pipeline_feature_74, "./randomforest.joblib", compress=True)

    return True
