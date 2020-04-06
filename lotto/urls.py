from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('prediction/', include('prediction.urls')),
    path('admin/', admin.site.urls),
]
