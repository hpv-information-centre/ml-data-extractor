from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='demo-home'),
    path('ajax/getArticle', views.getArticle, name="getArticle"),
    path('ajax/getArticlePredictions', views.getArticlePredictions, name="getArticlePredictions"),
]