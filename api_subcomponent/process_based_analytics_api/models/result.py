from threading import current_thread

from django.db import models

from rest_framework import serializers


class Result(models.Model):
    result_id = models.BigAutoField(primary_key=True)
    checkpoint = models.IntegerField()
    adaptation_action_trigger = models.BooleanField()
    prediction = models.FloatField()
    prediction_reliability = models.FloatField()
    explanation = models.TextField()
    case_id = models.ForeignKey('Case', on_delete=models.CASCADE)

    requests = {}

    @property
    def path(self):
        try:
            request = Result.requests[current_thread()]
        except KeyError:
            return "error fetching global request object for case " + str(self.result_id)
        return request.build_absolute_uri(
            "/datasets/" + str(self.case_id.dataset_id.dataset_id) + "/cases/" + str(
                self.case_id.case_id) + "/results/" + str(self.result_id) + "/")


class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = ['result_id', 'checkpoint', 'adaptation_action_trigger', 'prediction', 'prediction_reliability',
                  'explanation', 'case_id']


class ResultListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = ['result_id', 'checkpoint', 'adaptation_action_trigger', 'prediction', 'prediction_reliability',
                  'explanation', 'case_id', 'path']
