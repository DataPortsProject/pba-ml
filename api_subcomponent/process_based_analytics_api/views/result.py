from django.http import Http404
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.response import Response

from ..models import Case
from ..models.result import *
from ..util.common_functions import create_update, check_jwt_token


class ResultViewSet(viewsets.ViewSet):

    def list(self, request, case_id, **kwargs):
        check_jwt_token(request)
        Result.requests[current_thread()] = request
        queryset = Result.objects.filter(case_id=case_id)
        if queryset.count() == 0:
            raise Http404("There are no results for the case with id " + case_id)
        serializer = ResultListSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None, **kwargs):
        check_jwt_token(request)
        Result.requests[current_thread()] = request
        result = get_object_or_404(Result, result_id=pk)
        serializer = ResultSerializer(result)
        return Response(serializer.data)

    def create(self, request, case_id, **kwargs):
        check_jwt_token(request)
        Result.requests[current_thread()] = request
        serializer = ResultSerializer(data=request.data)
        case = get_object_or_404(Case, case_id=case_id)
        serializer.initial_data['case_id'] = case
        if 'result_id' in request.data:
            original = get_object_or_404(Result, result_id=request.data['result_id'])
            updated = serializer.update(original, serializer.initial_data)
            create_update(dataset=updated.case_id.dataset_id, case=updated.case_id, result=updated)
            return Response(ResultSerializer(updated).data)
        if serializer.is_valid():
            result = serializer.save()
            create_update(dataset=result.case_id.dataset_id, case=result.case_id, result=result)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
