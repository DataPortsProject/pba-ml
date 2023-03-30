from abc import ABC
from typing import Literal, List

import numpy as np
from predictive_process_monitoring.prbpm_models.dataset import Dataset


class BaseModelInterface(ABC):
    def get_feature_vector_length(self) -> int:
        return len(self.get_dataset().metadata)

    def get_feature_vector_labels(self) -> List[str]:
        return list(self.get_dataset().metadata.keys())

    def get_dataset(self) -> Dataset:
        """ Returns the Dataset.

        :rtype: Dataset.
        """
        raise NotImplementedError


class BinaryClassificationModelInterface(BaseModelInterface, ABC):
    def predict_binary_class_for_traces(self, traces: List[np.ndarray]) -> List[Literal[0, 1]]:
        """ Predicts classes for a batch of sequences.

        :param traces: List of traces. Each trace must be a Numpy array with shape (m, n), with m being
                       the number of events in the trace and n being the feature vector length.
        :rtype: A list of 0's and 1's, where each entry corresponds to the prediction of the corresponding input trace.
        """
        raise NotImplementedError


class BinaryDistributionModelInterface(BaseModelInterface, ABC):
   def predict_binary_probability_for_traces(self, traces: List[np.ndarray]) -> List[float]:
        """ Predicts probabilities for a batch of sequences.

        :param traces: List of traces. Each trace must be a Numpy array with shape (m, n), with m being
                       the number of events in the trace and n being the feature vector length.
        :rtype: A list of floats between 0 and 1, where each entry corresponds to the predicted probability that the
                class is 1 of the corresponding input trace.
        """
        raise NotImplementedError
