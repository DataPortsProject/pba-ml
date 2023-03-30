import logging

from django.core.management.base import BaseCommand

from process_based_analytics_api.models.dataset import *

logger = logging.getLogger("Test")


class Command(BaseCommand):
    help = 'Remove fake-data to the database'

    def handle(self, *args, **options):
        fake_dataset_names = ["BPIC12", "BPIC17", "cargo2000", "traffic"]
        for n in fake_dataset_names:
            queryset = Dataset.objects.filter(name=n)
            for dataset in queryset:
                dataset.delete()
                logger.info("deleted :" + str(dataset.name))
