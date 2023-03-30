from django.db import models
from rest_framework import serializers

from ..result import Result


class DisplayInformation(models.Model):
    result_id = models.OneToOneField(Result, on_delete=models.CASCADE, primary_key=True)
    ship_name = models.TextField()
    terminal_name = models.TextField()
    company = models.TextField()
    line = models.TextField()
    arrival = models.FloatField()
    departure = models.FloatField()
    status = models.TextField()


class DisplayInformationSerializer(serializers.ModelSerializer):
    class Meta:
        model = DisplayInformation
        fields = '__all__'
