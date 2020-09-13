from django.db import models
import uuid



from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
# Create your models here
class DataFrameToArrayTransformer(BaseEstimator,TransformerMixin):
  def fit(self,x,y=None):
    return self
  def transform(self,X):
    return np.transpose(np.matrix(X))
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class DB_model(models.Model):
    userName = models.CharField(max_length=100,null=True)
    userID  = models.UUIDField(primary_key = True,
         default = uuid.uuid4,
         editable = False)#CharField(max_length=100,null=True)
    text  = models.TextField()
    textLen = models.TextField()
    retweetsCount = models.TextField()
    favoriteCount = models.TextField()
    source =  models.TextField()
    language  = models.TextField()
    date = models.TextField()#DateTimeField(auto_now_add=True,auto_now=False)
    favourited = models.TextField()
    retweeted =models.TextField()
    userLocation= models.TextField()
    URL = models.URLField(max_length=200)
    userfollowers_count = models.TextField()
    userfriends_count= models.TextField()
    userListed_count= models.TextField()
    userFavorites_count= models.TextField()
    userStatuses_count= models.TextField()
    userVerified=models.TextField()
    userProtected=models.TextField()
    sentiment= models.IntegerField(default=-4)
    predictedlabel= models.IntegerField(default=-4)
    flag = models.IntegerField(default=-1)
    userHomelink=models.URLField(max_length=200, default='https://twitter.com/')
    user_profileImg = models.URLField(max_length=200, default='https://twitter.com/')
    topics=models.CharField(max_length=100, null=True,default="covid")
    twitterID = models.TextField()





