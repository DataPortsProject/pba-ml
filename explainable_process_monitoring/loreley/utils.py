import numpy as np


def get_number_of_different_values_for_feature(traces, feature_index) -> tuple[int, dict]:
    all_values_for_feature = [t[:, feature_index] for t in traces]  # list of ndarrays
    flattened_values_for_feature = np.concatenate(all_values_for_feature).ravel()
    flattened_without_nans = np.nan_to_num(flattened_values_for_feature, nan=-1)
    unique_values_for_feature = np.unique(flattened_without_nans)
    mapping_dict = dict()
    for index, value in enumerate(unique_values_for_feature):
        mapping_dict[value] = index
    return len(unique_values_for_feature), mapping_dict


def nan_safe_equals(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    if arr1.shape != arr2.shape:
        return False
    return ((arr1 == arr2) | (np.isnan([[float(x) for x in arr] for arr in arr1]) & np.isnan(
        [[float(x) for x in arr] for arr in arr2]))).all()


class Logger:

    def __init__(self, filename: str):
        self.filename = filename

    def log_in_file(self, log: str, clear: bool = False):
        mode = 'w' if clear else 'a'
        with open('results/' + self.filename + '.log', mode) as file:
            file.write(log + '\n')
        print(log)
