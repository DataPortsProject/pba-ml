import logging

from django.core.management.base import BaseCommand

from process_based_analytics_api.models.dataset import *
from ._private import test_dataset_names

logger = logging.getLogger("Test")


class Command(BaseCommand):
    help = 'Remove test-data from the database'

    def handle(self, *args, **options):
        for n in test_dataset_names:
            queryset = Dataset.objects.filter(name=n)
            for dataset in queryset:
                dataset.delete()
                logger.info("deleted :" + str(dataset.name))
