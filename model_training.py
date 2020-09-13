import pandas as pd
import numpy as np
import spacy
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import joblib
# Download a stopword list
import nltk


"""# Tokenization"""
nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')


nltk.download('stopwords')


# @Tokenize
def spacy_tokenize(string):
    tokens = list()
    doc = nlp(string)
    for token in doc:
        tokens.append(token)
    return tokens

# @Normalize
def normalize(tokens):
    normalized_tokens = list()
    for token in tokens:
        normalized = token.text.lower().strip()
        if ((token.is_alpha or token.is_digit)):
            normalized_tokens.append(normalized)
    return normalized_tokens


# @Tokenize and normalize
def tokenize_normalize(string):
    return normalize(spacy_tokenize(string))


def evaluation_summary(description, predictions, true_labels):
    print("Evaluation for: " + description)
    precision = precision_score(predictions, true_labels, average='macro')
    recall = recall_score(predictions, true_labels, average='macro')
    accuracy = accuracy_score(predictions, true_labels)
    f1 = fbeta_score(predictions, true_labels, 1, average='macro')  # 1 means f_1 measure
    print("Classifier '%s' has Acc=%0.3f P=%0.3f R=%0.3f F1=%0.3f" % (description, accuracy, precision, recall, f1))
    # Specify three digits instead of the default two.
    print(classification_report(predictions, true_labels, digits=3))
    print('\nConfusion matrix:\n',
          confusion_matrix(true_labels, predictions))  # Note the order here is true, predicted, odd.


class DataFrameToArrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # print(X.shape)
        # print(np.transpose(np.matrix(X)).shape)
        return np.transpose(np.matrix(X))


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""## Defining and initializing classifiers."""

one_hot_vectorizer = CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf1 = DummyClassifier(strategy="stratified")
# decisionTree= DecisionTreeClassifier(random_state=10)
#print('one hot vectorizer is ',one_hot_vectorizer)
dtaframe = pd.read_csv('/masterProject/project_files/temp_file/Dataset.csv', encoding='latin1')
firstcopy = secondcopy = dtaframe

#print(firstcopy.head(2))

y = (firstcopy['label'])

#print(len(y))


# nan values are replaced by this
firstcopy["source"].fillna("Twitter Web App", inplace=True)
firstcopy["userLocation"].fillna("No Location", inplace=True)
firstcopy["URL"].fillna("https://twitter.com/home", inplace=True)

firstcopy.dropna(inplace=True)
y.dropna(inplace=True)
firstcopy.drop(columns='Unnamed: 0', inplace=True)
firstcopy.drop(columns='label', inplace=True)


dataset1 = []
counter = 0
for index, row in firstcopy.iterrows():
    new_row = ""
    new_row = new_row + str(row['userName']) + " " + str(row['text']) + " " + str(row['textLen']) + " " + str(
        row['retweetsCount']) + " " + str(row['favoriteCount']) + " " + str(row['source']) + " " + str(
        row['language']) + " " + str(row['favourited']) + " " + str(row['retweeted']) + " " + str(
        row['userLocation']) + " " + str(row['URL']) + " " + str(row['userfollowers_count']) + " " + str(
        row['userfriends_count']) + " " + str(row['userListed_count']) + " " + str(
        row['userFavorites_count']) + " " + str(row['userStatuses_count']) + " " + str(row['userVerified']) + " " + str(
        row['userProtected']) + " " + str(row['sentiment'])
    dataset1.append(new_row)
    # print(type(new_row))
    counter = counter + 1
    # Val_labels.append(new_label)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

list_ = []

# Example code:
for sentence in firstcopy['text']:
    vs = analyzer.polarity_scores(sentence)
    list_.append(vs)
    # print("{:-<65} {}".format(sentence, str(vs)))

neg = pos = neu = compound = []
for i in list_:
    neg.append(i['neg'])
    pos.append(i['pos'])
    neu.append(i['neu'])
    compound.append(i['compound'])

data_frame = firstcopy

data_frame['negative'] = pd.DataFrame(neg, columns=['negative'])
data_frame['positive'] = pd.DataFrame(pos, columns=['positive'])
data_frame['compound'] = pd.DataFrame(compound, columns=['compound'])
data_frame['neutral'] = pd.DataFrame(neu, columns=['neutral'])

data_frame['spl'] = data_frame['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

import numpy as np
import warnings

# %matplotlib inline
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.corpus import stopwords

stop = stopwords.words('english')

data_frame['processedtext'] = data_frame['text'].str.replace('[^\w\s]', '')
data_frame['processedtext'] = data_frame['processedtext'].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop))
data_frame['processedtext'] = data_frame['processedtext'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Lines 4 to 6
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()
data_frame['processedtext'] = data_frame['processedtext'].apply(
    lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

"""# Adding new feature 2 
Entity
"""


stop = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")

df = []
entities = []
numOfEntities = []
for i in data_frame['processedtext']:
    df.append(nlp(i))
for i in df:
    sent = ''
    counter = 0
    for word in i.ents:
        counter = counter + 1
        sent = sent + " " + word.label_
        # print(word.text,word.label_)
    entities.append(sent)
    numOfEntities.append(counter)

data_frame['entities'] = pd.DataFrame(entities, columns=['entities'])
data_frame['numOfEntities'] = pd.DataFrame(numOfEntities, columns=['numOfEntities'])

data_frame.replace(r'^\s*$', "none", regex=True)

X_train22, X_test22, y_train22, y_test22 = train_test_split(
    data_frame, y, test_size=0.3, random_state=0)
X_train22, X_val22, y_train22, y_val22 = train_test_split(
    X_train22, y_train22, test_size=0.25, random_state=0)


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
print("\n RandomForest classifier Before feature added \n")
pipeline_feature_74.fit(X_train22,y_train22)
#result074 = pipeline_feature_74.predict(X_val22)
def predictions(df):
    model_=joblib.load("randomforest.joblib")
    return model_.predict(df)



joblib.dump(pipeline_feature_74, "./randomforest.joblib", compress=True)
#joblib.dump(one_hot_vectorizer,"./vectorizer.joblib", compress=True)

x = data_frame.head(3)
result= predictions(x)
print('result is === ',result)
#joblib.dump(result,"./resutl.joblib", compress=True)