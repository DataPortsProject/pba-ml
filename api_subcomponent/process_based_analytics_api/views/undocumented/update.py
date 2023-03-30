from ...models.undocumented.update import Update, UpdateSerializer
from ...util.common_functions import check_jwt_token

from threading import current_thread

from django.http import HttpResponse
from django.shortcuts import get_object_or_404

from rest_framework import generics, status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response


class UpdateViewSet(viewsets.ViewSet):

    def list(self, request, update_tracker_id, **kwargs):
        check_jwt_token(request)
        queryset = Update.objects.filter(update_tracker_id=update_tracker_id)
        serializer = UpdateSerializer(queryset, many=True)
        response = Response(serializer.data)
        queryset.delete()
        return response


