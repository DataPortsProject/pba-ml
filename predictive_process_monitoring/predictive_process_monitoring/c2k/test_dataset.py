from __future__ import annotations

from typing import Literal

import math
import numpy as np
import pandas as pd
from predictive_process_monitoring.prbpm_models.dataset import Dataset, Batch, PackedBatch, pack_batch, unpack_batch
import pickle


class Cargo2000(Dataset):

    @staticmethod
    def delimiter():
        return ';'

    @staticmethod
    def categorical_columns():
        return ['ActivityID', 'AirportCode']

    @staticmethod
    def numerical_columns():
        return ['Duration', 'TimeSinceLastEvent', 'Timestamp', 'PlannedDuration',
                               'PlannedTimestamp', 'PlannedEndTimestamp']

    @staticmethod
    def case_attribute_columns():
        return ['PlannedEndTimestamp']

    @staticmethod
    def timestamp_columns():
        return ['Timestamp']

    @staticmethod
    def delete_columns():
        return ['CaseID', 'InstanceID', 'EndTimestamp']

    @staticmethod
    def trace_column_name():
        return 'CaseID'

    @staticmethod
    def _get_trace_label(trace: pd.DataFrame) -> Literal[0, 1, None]:
        is_violation = trace.iloc[0]['PlannedEndTimestamp'] < trace.iloc[0]['EndTimestamp']
        return 1 if is_violation else 0
