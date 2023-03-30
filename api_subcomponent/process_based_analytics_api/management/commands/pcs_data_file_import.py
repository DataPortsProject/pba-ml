import logging
import os
import random

import pandas
from django.core.management import BaseCommand

from ._private import test_result_explanations
from ...models import Case, Result, DisplayInformation
from ...models.dataset import Dataset

logger = logging.getLogger("Test")


class Command(BaseCommand):

    def handle(self, *args, **options):
        queryset = Dataset.objects.filter(name="PCS")
        if queryset.count() != 0:
            logger.info(f"PCS data already imported")
            return

        dataframe = pandas.read_csv(os.path.join("database", "VP_clean.csv"), sep=",", header=0, decimal=".",
                                    quotechar="\"")
        amount_rows = len(dataframe.index)
        amount_rows = min(amount_rows, 300)

        dataset = Dataset(name="PCS")
        dataset.save()

        previous_ship_name = ""
        checkpoint = 1
        case = None
        for index, row in dataframe.iterrows():
            ship_name = row["Buque"]
            if ship_name != previous_ship_name:
                checkpoint = 1
                previous_ship_name = ship_name
                last_50_rows = amount_rows - 50 <= index
                case = Case(dataset_id=dataset, open=last_50_rows)
                case.save()

            result = Result(checkpoint=checkpoint, adaptation_action_trigger=(index % 10 == 0),
                            prediction=random.random(),
                            prediction_reliability=random.random(), case_id=case,
                            explanation=random.choice(test_result_explanations))
            result.save()

            status = row["Estado"]
            if status == "Prevista":
                status = "announced"
            elif status == "Autorizada":
                status = "authorized"
            elif status == "Operando":
                status = "operational"
            elif status == "Finalizada":
                status = "finilization"

            display_information = DisplayInformation(result_id=result, ship_name=ship_name,
                                                     terminal_name=row["Terminal"], company=row["Consignatario buque"],
                                                     line=row["Línea regular"], arrival=row["Llegada"],
                                                     departure=row["Salida"], status=status)
            display_information.save()

            checkpoint += 1

            if index % 33 == 0:
                logger.info(f"progress {index / 33}/{round(amount_rows / 33)}")

            if index >= amount_rows:
                break

        logger.info(f"done importing PCS data")
