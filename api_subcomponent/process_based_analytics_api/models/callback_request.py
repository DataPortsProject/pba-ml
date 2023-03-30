from django.db import models
from rest_framework import serializers


class CallbackRequest(models.Model):
    callback_request_id = models.AutoField(primary_key=True)
    subscribed_to_dataset_id = models.IntegerField(null=True)
    subscribed_to_case_id = models.IntegerField(null=True)
    callback_address = models.URLField()


class CallbackRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = CallbackRequest
        fields = '__all__'
