from django.urls import path
from detect import views


urlpatterns = [
    path('', views.index),
    path('<Position>/', views.video_stream),
]
