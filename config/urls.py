from django.contrib import admin
from django.urls import path,include
from detect import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detect/', include('detect.urls')),
    path('get_data/',views.get_data)


]
