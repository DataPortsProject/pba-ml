import math
from typing import Literal, List

import numpy as np
from predictive_process_monitoring.prbpm_models.dataset import Dataset, Batch
from predictive_process_monitoring.prbpm_models.interface import BinaryClassificationModelInterface, BinaryDistributionModelInterface
from predictive_process_monitoring.prbpm_models.model import EnsembleModel


class InferenceModel(BinaryClassificationModelInterface, BinaryDistributionModelInterface):
    def get_dataset(self) -> Dataset:
        return self.model.dataset

    def predict_binary_class_for_traces(self, traces: List[np.ndarray]) -> List[Literal[0, 1]]:
        batch_probabilities = self.predict_binary_probability_for_traces(traces)
        return [1 if probability > 0.5 else 0 for probability in batch_probabilities]

    def predict_binary_probability_for_traces(self, traces: List[np.ndarray]) -> List[float]:
        return self.model.predict_probabilities(Batch(traces, self.model.dataset.metadata))

    def __init__(self, ensemble_model: EnsembleModel):
        self.model: EnsembleModel = ensemble_model
