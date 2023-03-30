from threading import current_thread

from rest_framework import viewsets
from rest_framework.response import Response

from ..models.case import Case, CaseListSerializer
from ..util.common_functions import check_jwt_token


class OpenCasesViewSet(viewsets.ViewSet):

    def list(self, request, dataset_id):
        check_jwt_token(request)
        Case.requests[current_thread()] = request
        queryset = Case.objects.filter(dataset_id=dataset_id, open=True)
        serializer = CaseListSerializer(queryset, many=True)
        return Response(serializer.data)
