from django.db import models
from rest_framework import serializers
from threading import current_thread


class UpdateTracker(models.Model):
    update_tracker_id = models.AutoField(primary_key=True)
    source = models.TextField(default="")

    requests = {}

    @property
    def updates(self):
        try:
            request = UpdateTracker.requests[current_thread()]
        except KeyError:
            return "error fetching global request object for dataset " + str(self.update_tracker_id)
        return request.build_absolute_uri("/update_tracker/" + str(self.update_tracker_id) + "/updates/")


class UpdateTrackerSerializer(serializers.ModelSerializer):
    class Meta:
        model = UpdateTracker
        fields = ['update_tracker_id', 'source', 'updates']
