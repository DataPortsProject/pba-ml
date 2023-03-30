import logging

import jwt
import requests

from django_api_server import settings
from ..models import CallbackRequest
from ..models.case import Case
from ..models.dataset import Dataset
from ..models.result import Result
from ..models.undocumented.update import UpdateSerializer
from ..models.undocumented.update_tracker import UpdateTracker


def send_callbacks(update_data: dict):
    queryset = CallbackRequest.objects.all()
    for callback_request in queryset:
        everything = callback_request.subscribed_to_dataset_id is None and callback_request.subscribed_to_case_id is None
        relevant_dataset = callback_request.subscribed_to_dataset_id == update_data['dataset_id'] and (
                callback_request.subscribed_to_case_id is None or callback_request.subscribed_to_case_id ==
                update_data['case_id'])
        relevant_case = callback_request.subscribed_to_case_id == update_data['case_id']
        if everything or relevant_case or relevant_dataset:
            try:
                requests.post(callback_request.callback_address, json=update_data)
            except:
                logging.exception("Exception occurred while sending callback requests!")


def create_update(dataset: Dataset = None, case: Case = None, result: Result = None):
    queryset = UpdateTracker.objects.all()

    def _construct_update_serializer(dataset: Dataset = None, case: Case = None, result: Result = None):
        serializer = UpdateSerializer(data={})
        if dataset is not None:
            serializer.initial_data['dataset_id'] = dataset.dataset_id
        if case is not None:
            serializer.initial_data['case_id'] = case.case_id
        if result is not None:
            serializer.initial_data['result_id'] = result.result_id
        return serializer

    for tracker in queryset:
        serializer = _construct_update_serializer(dataset=dataset, case=case, result=result)
        serializer.initial_data['update_tracker_id'] = tracker.update_tracker_id
        if serializer.is_valid(raise_exception=True):
            serializer.save()

    update_data = _construct_update_serializer(dataset=dataset, case=case, result=result).initial_data
    send_callbacks(update_data=update_data)


def check_jwt_token(request):
    if settings.DEBUG:
        return True

    if "HTTP_AUTHORIZATION" not in request.META:
        raise ValueError("No authorization token provided!")
    token_canditates = request.META["HTTP_AUTHORIZATION"].split("Bearer ")
    token_canditates.remove("")
    token = token_canditates[0]

    payload = jwt.decode(
        token,
        algorithms=["RS256", "HS256"], options={"verify_signature": False})
    return payload is not None


def check_display_jwt_token(request):
    if settings.DEBUG:
        return True

    if "HTTP_AUTHORIZATION" not in request.META:
        raise ValueError("No authorization token provided!")
    token_canditates = request.META["HTTP_AUTHORIZATION"].split("Bearer ")
    token_canditates.remove("")
    token = token_canditates[0]

    payload = jwt.decode(
        token, "13375h17219h7h323",
        algorithms=["HS256", "RS256"])
    return "azp" in payload and payload["azp"] == "PBA-UI"
