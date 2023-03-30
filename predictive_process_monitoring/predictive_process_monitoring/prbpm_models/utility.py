import imp
import numpy as np
import math


def zero_pad_traces(traces: list[np.ndarray], prefix=True, max_length=None) -> list[np.ndarray]:
    """
    Zero pad all traces to match a length of max_length.
    If max_length is None the longest prefix is used.
    """

    def get_zero_padded_trace(trace, length) -> np.ndarray:
        if len(trace) == length:
            return trace

        feature_vector_length = len(trace[0])
        zero_padding_length = length - len(trace)
        zero_padding = [[0] * feature_vector_length] * zero_padding_length
        return np.insert(trace, 0, zero_padding, axis=0) if prefix else np.insert(trace, -1, zero_padding, axis=0)

    if max_length is None:
        max_length = max([len(trace) for trace in traces])

    zero_padded_traces = [get_zero_padded_trace(trace, max_length) for trace in traces]
    return zero_padded_traces


def ngram_traces(traces: list[np.ndarray], n=None) -> list[np.ndarray]:
    """
    Create n-grams from all traces to match a length of n.
    If n is None the longest trace is used.
    """

    if n is None:
        n = max([len(trace) for trace in traces])

    ngrammed_traces = []
    for trace in traces:
        ngrammed_traces.extend(ngram_trace(trace, n))  # extend as ngram_trace gives list of traces

    return ngrammed_traces


def ngram_trace(trace: np.ndarray, n) -> list[np.ndarray]:
    n_subtraces = [trace[max(0, i-n):i+1] for i in range(len(trace))]
    ngrams = zero_pad_traces(n_subtraces, max_length=n)
    return ngrams


def index_in_list_nan_robust(haystack, needle):
    if type(needle) is str or not math.isnan(needle):
        return haystack.index(needle)

    for i, element in enumerate(haystack):
        if type(element) is not str and math.isnan(element):
            return i

    raise ValueError('nan is not in list')


def one_hot_encode(column: np.ndarray, categorical_values: list) -> np.ndarray:
    one_hot_encoded = np.zeros((len(column), len(categorical_values)))
    one_column_indices = [index_in_list_nan_robust(categorical_values, val) for val in column]
    one_hot_encoded[(np.arange(len(column)), one_column_indices)] = 1.0
    return one_hot_encoded
