import logging
import random

from django.core.management.base import BaseCommand
from django.db.transaction import atomic

from process_based_analytics_api.models.dataset import *
from process_based_analytics_api.models.result import *
from ._private import test_dataset_names, test_result_explanations, test_case_amount

logger = logging.getLogger("Test")


class Command(BaseCommand):
    help = 'Add test-data to the database if it is not already there'

    def handle(self, *args, **options):
        for n in test_dataset_names:
            queryset = Dataset.objects.filter(name=n)
            if queryset.count() == 0:
                dataset = Dataset(name=n)
                dataset.save()
                logger.info("created :" + str(dataset.name))
                cases = self._create_cases(dataset)
                logger.info("created all cases")
                for i in range(len(cases)):
                    self._create_results(cases[i])
                    if i % 100 == 0:
                        logger.info("case: " + str(cases[i].case_id))

    @atomic
    def _create_cases(self, dataset):
        result = []
        for i in range(max(test_case_amount-10, 0)):
            case = Case(open=False, dataset_id=dataset)
            case.save()
            result.append(case)
        for i in range(min(test_case_amount, 10)):
            case = Case(open=True, dataset_id=dataset)
            case.save()
            result.append(case)
        return result

    @atomic()
    def _create_results(self, case):
        for i in range(int(random.random() * 100 + 1)):
            result = Result(checkpoint=(i + 1), adaptation_action_trigger=(i % 10 == 0), prediction=random.random(),
                            prediction_reliability=random.random(), case_id=case,
                            explanation=random.choice(test_result_explanations))
            result.save()
