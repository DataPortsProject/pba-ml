from threading import current_thread

from django.http import Http404
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.response import Response

from ...models.undocumented.update_tracker import UpdateTracker, UpdateTrackerSerializer
from ...util.common_functions import check_jwt_token


class UpdateTrackerViewSet(viewsets.ViewSet):

    def list(self, request):
        check_jwt_token(request)
        UpdateTracker.requests[current_thread()] = request
        queryset = UpdateTracker.objects.all()
        if queryset.count() == 0:
            s = status.HTTP_204_NO_CONTENT
        else:
            s = status.HTTP_200_OK
        serializer = UpdateTrackerSerializer(queryset, many=True)
        return Response(serializer.data, status=s)

    def retrieve(self, request, pk=None):
        UpdateTracker.requests[current_thread()] = request
        update_tracker = get_object_or_404(UpdateTracker, update_tracker_id=pk)
        serializer = UpdateTrackerSerializer(update_tracker)
        return Response(serializer.data)

    def create(self, request):
        UpdateTracker.requests[current_thread()] = request
        serializer = UpdateTrackerSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        UpdateTracker.requests[current_thread()] = request
        update_tracker = get_object_or_404(UpdateTracker, update_tracker_id=pk)
        update_tracker.delete()
        return Response()
