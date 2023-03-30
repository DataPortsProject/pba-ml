"""django_api_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path, re_path
from rest_framework import routers

from django_api_server import settings
from process_based_analytics_api.views.callback_request import *
from process_based_analytics_api.views.case import *
from process_based_analytics_api.views.dataset import *
from process_based_analytics_api.views.open_cases import *
from process_based_analytics_api.views.result import *
from process_based_analytics_api.views.undocumented.display_information import *
from process_based_analytics_api.views.undocumented.update import *
from process_based_analytics_api.views.undocumented.update_tracker import *

if settings.DEBUG:
    router = routers.DefaultRouter()
else:
    router = routers.SimpleRouter()
router.register(r'datasets', DatasetViewSet, 'dataset')
router.register(r'datasets/(?P<dataset_id>\d+)/cases', CaseViewSet, 'case')
router.register(r'datasets/(?P<dataset_id>\d+)/open_cases', OpenCasesViewSet, 'open_cases')
router.register(r'datasets/(?P<dataset_id>\d+)/cases/(?P<case_id>\d+)/results', ResultViewSet, 'result')
router.register(r'callback/outgoing', CallbackRequestViewSet, 'callback')
router.register(r'update_tracker', UpdateTrackerViewSet, 'update_tracker')
router.register(r'update_tracker/(?P<update_tracker_id>\d+)/updates', UpdateViewSet, 'updates')

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^datasets/(?P<dataset_id>\d+)/cases/(?P<case_id>\d+)/results/(?P<result_id>\d+)/display_information',
            DisplayInformationView.as_view()),
    re_path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]
