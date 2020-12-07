from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path('', views.Index.as_view())
]


urlpatterns+=static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns+=static(settings.IMAGE_URL, document_root=settings.IMAGE_ROOT)
urlpatterns+=static(settings.PRED_URL, document_root=settings.PRED_ROOT)
urlpatterns+=static(settings.PREPROC_URL, document_root=settings.PREPROC_ROOT)