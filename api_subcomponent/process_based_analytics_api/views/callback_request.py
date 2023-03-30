from django.http import Http404
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.response import Response

from ..models.callback_request import CallbackRequest, CallbackRequestSerializer
from ..util.common_functions import check_jwt_token


class CallbackRequestViewSet(viewsets.ViewSet):

    def list(self, request):
        check_jwt_token(request)
        queryset = CallbackRequest.objects.all()
        if queryset.count() == 0:
            s = status.HTTP_204_NO_CONTENT
        else:
            s = status.HTTP_200_OK
        serializer = CallbackRequestSerializer(queryset, many=True)
        return Response(serializer.data, status=s)

    def retrieve(self, request, pk=None):
        check_jwt_token(request)
        callback_request = get_object_or_404(CallbackRequest, callback_request_id=pk)
        serializer = CallbackRequestSerializer(callback_request)
        return Response(serializer.data)

    def create(self, request):
        check_jwt_token(request)
        serializer = CallbackRequestSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        check_jwt_token(request)
        callback_request = get_object_or_404(CallbackRequest, callback_request_id=pk)
        callback_request.delete()
        return Response()
