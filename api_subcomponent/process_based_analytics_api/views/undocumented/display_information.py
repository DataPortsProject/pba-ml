from django.http import Http404
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from ...models.undocumented.display_information import DisplayInformation, DisplayInformationSerializer
from ...util.common_functions import check_display_jwt_token, create_update


class DisplayInformationView(APIView):

    def get(self, *args, **kwargs):
        result_id = self.kwargs["result_id"]
        request = self.request
        check_display_jwt_token(request)
        queryset = DisplayInformation.objects.filter(result_id=result_id)
        if queryset.count() == 0:
            raise Http404(f"There is no display_information for result {result_id}")
        serializer = DisplayInformationSerializer(queryset[0])
        return Response(serializer.data)

    def post(self, *args, **kwargs):
        result_id = self.kwargs["result_id"]
        request = self.request
        check_display_jwt_token(request)
        serializer = DisplayInformationSerializer(data=request.data)
        queryset = DisplayInformation.objects.filter(result_id=result_id)
        if queryset.count() != 0:
            original = queryset[0]
            updated = serializer.update(original, serializer.initial_data)
            create_update(dataset=updated.result_id.case_id.dataset_id, case=updated.result_id.case_id,
                          result=updated.result_id)
            return Response(DisplayInformationSerializer(updated).data)
        else:
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
