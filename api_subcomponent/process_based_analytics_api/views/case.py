from django.http import Http404
from django.shortcuts import get_object_or_404
from rest_framework import generics, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ..models.callback_request import CallbackRequestSerializer
from ..models.case import *
from ..util.common_functions import create_update, check_jwt_token


class CaseViewSet(viewsets.ViewSet, generics.GenericAPIView):

    def get_serializer_class(self):
        return CaseSerializer

    def get_queryset(self):
        return Case.objects.all()

    def list(self, request, dataset_id):
        check_jwt_token(request)
        Case.requests[current_thread()] = request
        queryset = Case.objects.filter(dataset_id=dataset_id)
        if queryset.count() == 0:
            raise Http404("There are no cases in the dataset with id " + dataset_id)
        page = self.paginate_queryset(queryset)
        serializer = CaseListSerializer(page, many=True)
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)

    def retrieve(self, request, pk=None, **kwargs):
        check_jwt_token(request)
        Case.requests[current_thread()] = request
        case = get_object_or_404(Case, case_id=pk)
        serializer = CaseSerializer(case)
        return Response(serializer.data)

    def create(self, request, dataset_id):
        check_jwt_token(request)
        Case.requests[current_thread()] = request
        serializer = CaseSerializer(data=request.data)
        serializer.initial_data['dataset_id'] = dataset_id
        if serializer.is_valid():
            case = serializer.save()
            create_update(dataset=case.dataset_id, case=case)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def subscribe(self, request, pk=None, **kwargs):
        check_jwt_token(request)
        Case.requests[current_thread()] = request
        serializer = CallbackRequestSerializer(data=request.data)
        serializer.initial_data['subscribed_to_case_id'] = pk
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def close(self, request, pk=None, **kwargs):
        check_jwt_token(request)
        Case.requests[current_thread()] = request
        case = get_object_or_404(Case, case_id=pk)
        case.open = False
        case.save()
        create_update(dataset=case.dataset_id, case=case)
        return Response(status=status.HTTP_200_OK)
