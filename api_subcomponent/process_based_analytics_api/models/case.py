from .result import Result, ResultListSerializer

from threading import current_thread

from django.db import models

from rest_framework import serializers


class Case(models.Model):
    case_id = models.BigAutoField(primary_key=True)
    open = models.BooleanField(db_index=True)
    dataset_id = models.ForeignKey('Dataset', on_delete=models.CASCADE)

    requests = {}

    @property
    def total_checkpoints(self):
        queryset = Result.objects.filter(case_id=self.case_id)
        return queryset.count()

    @property
    def results(self):
        try:
            request = Case.requests[current_thread()]
        except KeyError:
            return "error fetching global request object for case " + str(self.case_id)
        return request.build_absolute_uri(
            "/datasets/" + str(self.dataset_id.dataset_id) + "/cases/" + str(self.case_id) + "/results/")

    @property
    def path(self):
        try:
            request = Case.requests[current_thread()]
        except KeyError:
            return "error fetching global request object for case " + str(self.case_id)
        return request.build_absolute_uri(
            "/datasets/" + str(self.dataset_id.dataset_id) + "/cases/" + str(self.case_id) + "/")


class CaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Case
        fields = ['case_id', 'open', 'total_checkpoints', 'dataset_id', 'results']


class CaseListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Case
        fields = ['case_id', 'open', 'total_checkpoints', 'dataset_id', 'results', 'path']
