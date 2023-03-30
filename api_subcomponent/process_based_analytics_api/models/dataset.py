from .case import Case, CaseListSerializer
from .result import Result

from threading import current_thread

from django.db import models

from rest_framework import serializers


class Dataset(models.Model):
    dataset_id = models.BigAutoField(primary_key=True)
    name = models.TextField()

    requests = {}

    @property
    def total_cases(self):
        queryset = Case.objects.filter(dataset_id=self.dataset_id)
        return queryset.count()

    @property
    def min_case_length(self):
        # Waaaaaaay too slow, all this needs to happen at the database level,
        # but I haven't figured out how to do that with Django ORM yet
        queryset = Case.objects.filter(dataset_id=self.dataset_id)
        result = 0
        for c in queryset:
            if result == 0 or c.total_checkpoints < result:
                result = c.total_checkpoints
        return result

    @property
    def max_case_length(self):
        # Waaaaaaay too slow, all this needs to happen at the database level,
        # but I haven't figured out how to do that with Django ORM yet
        queryset = Case.objects.filter(dataset_id=self.dataset_id)
        result = 0
        for c in queryset:
            if result == 0 or c.total_checkpoints > result:
                result = c.total_checkpoints
        return result

    @property
    def cases(self):
        try:
            request = Dataset.requests[current_thread()]
        except KeyError:
            return "error fetching global request object for dataset " + str(self.dataset_id)
        return request.build_absolute_uri("/datasets/" + str(self.dataset_id) + "/cases/")

    @property
    def path(self):
        try:
            request = Dataset.requests[current_thread()]
        except KeyError:
            return "error fetching global request object for dataset " + str(self.dataset_id)
        return request.build_absolute_uri("/datasets/" + str(self.dataset_id) + "/")

    @property
    def open_cases(self):
        try:
            request = Dataset.requests[current_thread()]
        except KeyError:
            return "error fetching global request object for dataset " + str(self.dataset_id)
        return request.build_absolute_uri("/datasets/" + str(self.dataset_id) + "/open_cases/")


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['dataset_id', 'name', 'total_cases', 'cases', 'open_cases']


class DatasetListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['dataset_id', 'name', 'total_cases', 'cases', 'open_cases', 'path']
