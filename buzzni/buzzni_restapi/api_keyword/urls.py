from django.urls import path
from . import views

app_name = 'api_keyword'
urlpatterns = [
    path('', views.KeywordView.as_view()),
]