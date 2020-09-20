from django.contrib import admin
from django.urls import path
from . import views
import spacy
import en_core_web_sm
urlpatterns = [
    path('', views.home, name = 'home'),
    path('ser', views.search, name='ser'),
    path('radiobtn',views.radiobtn,name="temp.html"),
    path('back', views.back, name="homepage.html"),
   # path('retrain',views.retrain, name='retraining'),
]
