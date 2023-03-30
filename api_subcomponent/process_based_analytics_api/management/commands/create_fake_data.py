import logging
import math
import random

from django.core.management.base import BaseCommand
from django.db.transaction import atomic

from process_based_analytics_api.models.dataset import *
from process_based_analytics_api.models.result import *

logger = logging.getLogger("Test")


class Command(BaseCommand):
    help = 'Add fake-data to the database if it is not already there'

    def handle(self, *args, **options):
        self._create_fake_data()

    @atomic
    def _create_fake_data(self):
        bpic12 = Dataset(name="BPIC12")
        bpic12.save()
        self._create_fake_cases(bpic12, 13087, 23)
        bpic17 = Dataset(name="BPIC17")
        bpic17.save()
        self._create_fake_cases(bpic17, 31413, 23)
        cargo2k = Dataset(name="cargo2000")
        cargo2k.save()
        self._create_fake_cases(cargo2k, 3942, 7)
        traffic = Dataset(name="traffic")
        traffic.save()
        self._create_fake_cases(traffic, 129615, 4)

    def _create_fake_cases(self, dataset, cases, checkpoints):
        for c in range(cases):
            case = Case(open=False, dataset_id=dataset)
            case.save()
            for r in range(checkpoints):
                if r < checkpoints / 2 or random.random() > 0.1:
                    result = Result(checkpoint=(r + 1), adaptation_action_trigger=False,
                                    prediction=0.5 + random.random() / 2,
                                    prediction_reliability=0.5 + math.pow(random.random(), min(3, checkpoints - r)) / 2,
                                    case_id=case,
                                    explanation=random.choice("-"))
                else:
                    result = Result(checkpoint=(r + 1), adaptation_action_trigger=True,
                                    prediction=1.1 + random.random() / 2,
                                    prediction_reliability=0.5 + math.pow(random.random(), 0.25) / 2,
                                    case_id=case,
                                    explanation="(v[1,AirportCode]=nan)≠420.0")
                    result.save()
                    break
                result.save()
