from threading import current_thread

from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ..models.callback_request import CallbackRequestSerializer
from ..models.dataset import Dataset, DatasetSerializer, DatasetListSerializer
from ..util.common_functions import create_update, check_jwt_token


class DatasetViewSet(viewsets.ViewSet):

    def list(self, request):
        check_jwt_token(request)
        Dataset.requests[current_thread()] = request
        queryset = Dataset.objects.all()
        serializer = DatasetListSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        check_jwt_token(request)
        Dataset.requests[current_thread()] = request
        queryset = Dataset.objects.all()
        dataset = get_object_or_404(queryset, dataset_id=pk)
        serializer = DatasetSerializer(dataset)
        return Response(serializer.data)

    def create(self, request):
        check_jwt_token(request)
        Dataset.requests[current_thread()] = request
        serializer = DatasetSerializer(data=request.data)
        if serializer.is_valid():
            dataset = serializer.save()
            create_update(dataset=dataset)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        check_jwt_token(request)
        dataset = get_object_or_404(Dataset, dataset_id=pk)
        dataset.delete()
        create_update(dataset=dataset)
        return Response()

    @action(detail=True, methods=['post'])
    def subscribe(self, request, pk=None):
        check_jwt_token(request)
        Dataset.requests[current_thread()] = request
        serializer = CallbackRequestSerializer(data=request.data)
        serializer.initial_data['subscribed_to_dataset_id'] = pk
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
