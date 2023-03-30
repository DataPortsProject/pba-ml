from django.db import models
from rest_framework import serializers


class Update(models.Model):
    update_id = models.AutoField(primary_key=True)
    update_tracker_id = models.ForeignKey('UpdateTracker', on_delete=models.CASCADE)
    dataset_id = models.ForeignKey('Dataset', on_delete=models.DO_NOTHING, null=True)
    case_id = models.ForeignKey('Case', on_delete=models.DO_NOTHING, null=True)
    result_id = models.ForeignKey('Result', on_delete=models.DO_NOTHING, null=True)


class UpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Update
        fields = '__all__'
