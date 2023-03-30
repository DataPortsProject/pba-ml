from __future__ import annotations

from abc import ABC
from copy import deepcopy
import pandas as pd

import numpy as np
from predictive_process_monitoring.prbpm_models.utility import one_hot_encode, zero_pad_traces, ngram_traces, ngram_trace
from typing import Generator
from collections import OrderedDict
from typing import Literal

import pickle


def ngram_dataset(dataset: Dataset, n=None):
    batch, labels = dataset.dataset

    if n is None:
        n = max([len(trace) for trace in batch.traces])

    ngrammed_traces_lists = [ngram_trace(trace, n) for trace in batch.traces]

    labels_for_ngrams = []
    flattened_ngrammed_traces = []
    for i, ngrammed_traces in enumerate(ngrammed_traces_lists):
        # for each ngrammed trace we need the corresponding label
        labels_for_ngrams.extend([labels[i]] * len(ngrammed_traces))
        flattened_ngrammed_traces.extend(ngrammed_traces)

    return Dataset(flattened_ngrammed_traces, labels_for_ngrams, deepcopy(dataset.metadata))


def pre_zero_pad_dataset(dataset: Dataset, extend_labels=False) -> Batch:
    batch, labels = dataset.dataset

    zero_padded_batch = pre_zero_pad(batch)

    longest_trace_length = max([len(trace) for trace in zero_padded_batch.traces])
    zero_padded_labels = [[[0] if all([a == 0 for a in zero_padded_batch.traces[i][timestep]]) else [val] for timestep
             in range(longest_trace_length)] for i, val in enumerate(labels)]  # duplicate labels

    return Dataset(zero_padded_batch.traces, zero_padded_labels if extend_labels else labels, deepcopy(dataset.metadata))


def ngram_batch(batch: Batch, n=None) -> Batch:  # n of None equals length of longest trace in batch
    return Batch(ngram_traces(batch.traces, n), deepcopy(batch.metadata))


def pre_zero_pad(batch: Batch) -> Batch:
    return Batch(zero_pad_traces(batch.traces), deepcopy(batch.metadata))


def post_zero_pad(batch: Batch) -> Batch:
    return Batch(zero_pad_traces(batch.traces, prefix=False), deepcopy(batch.metadata))


def one_hot_encode_columns(batch: Batch, columns: list[str]) -> Batch:
    for column in columns:
        batch = one_hot_encode_column(batch, column)

    return batch

def one_hot_encode_all_columns(batch: Batch) -> Batch:
    return one_hot_encode_columns(batch, [k for k,v in batch.metadata.items() if v['type']=='categorical'])

def one_hot_encode_column(batch: Batch, column: str) -> Batch:
    assert batch.metadata[column]['type'] == 'categorical', 'Can only one hot encode categorical columns.'
    target_column_index = batch.metadata[column]['index']
    unique_values = batch.metadata[column]['unique_values']

    one_hot_encoded_traces = []
    for trace in batch.traces:
        target_column = trace[:, target_column_index]
        one_hot_encoded_column = one_hot_encode(target_column, unique_values)  # len(unique_values) different columns
        encoded_trace = np.delete(trace, target_column_index, axis=1)
        encoded_trace = np.append(encoded_trace, one_hot_encoded_column, axis=1)
        one_hot_encoded_traces.append(encoded_trace)

    number_of_one_hot_columns = len(unique_values)
    one_hot_encoded_index_offset = len(one_hot_encoded_traces[0][0]) - number_of_one_hot_columns
    new_metadata = _delete_column_from_metadata(batch.metadata, column)
    new_metadata[column] = {
        'type': 'one_hot',
        'is_case_attribute': batch.metadata[column]['is_case_attribute'],
        'value_index': {val: one_hot_encoded_index_offset + i for i, val in enumerate(unique_values)}
    }

    return Batch(one_hot_encoded_traces, new_metadata)


def _decrement_index_in_metadata(metadata: dict, only_decrement_greater_than: int) -> dict:
    new_metadata = deepcopy(metadata)
    for column_name, info in new_metadata.items():  # all columns after the one deleted now have ID - 1
        if info['type'] in ('categorical', 'numerical') and info['index'] > only_decrement_greater_than:
            info['index'] -= 1
        elif info['type'] == 'one_hot':
            for key, column_index in info['value_index'].items():
                if column_index > only_decrement_greater_than:
                    info['value_index'][key] -= 1

    return new_metadata


def _delete_column_from_metadata(metadata: dict, column: str) -> dict:
    new_metadata = deepcopy(metadata)

    if metadata[column]['type'] in ('categorical', 'numerical'):
        delete_column_index = metadata[column]['index']
        new_metadata = _decrement_index_in_metadata(new_metadata, only_decrement_greater_than=delete_column_index)

    elif metadata[column]['type'] == 'one_hot':
        delete_column_indices = metadata[column]['value_index'].values()
        delete_column_indices = sorted(delete_column_indices, reverse=True)  # we have to start with greatest index
        for index in delete_column_indices:
            new_metadata = _decrement_index_in_metadata(new_metadata, only_decrement_greater_than=index)

    del new_metadata[column]
    return new_metadata


def one_hot_to_categorical(batch: Batch, column: str) -> Batch:
    assert batch.metadata[column]['type'] == 'one_hot', 'Can only convert one hot encoded columns.'
    value_indices = batch.metadata[column]['value_index']

    decoded_traces = []
    for trace in batch.traces:
        categorical_column = [value for value, column_index in value_indices.items()
                              for event in trace if event[column_index] == 1]

        # delete all one hot columns
        one_hot_column_ids = tuple(value_indices.values())
        trace_without_one_hot = np.delete(trace, one_hot_column_ids, axis=1)
        decoded_trace = np.append(trace_without_one_hot, np.transpose([categorical_column]), axis=1)
        decoded_traces.append(decoded_trace)

    new_feature_vector_length = len(decoded_traces[0][0])
    new_metadata = _delete_column_from_metadata(batch.metadata, column)
    new_metadata[column] = {
        'type': 'categorical',
        'is_case_attribute': batch.metadata[column]['is_case_attribute'],
        'index': new_feature_vector_length - 1,  # appended -> added as last column
        'unique_values': tuple(value_indices.keys())
    }

    return Batch(decoded_traces, new_metadata)

def mask_nan_all_columns(batch: Batch) -> Batch:
    nan_columns = []
    for column, metadata in [(k,v) for k,v in batch.metadata.items() if v['type']=='numerical']:
        for trace in batch.traces:
            for event in trace:
                if np.isnan(event[metadata['index']]):
                    break
            else:
                continue
            break
        else:
            continue
        nan_columns.append(column)
    return mask_nan_columns(batch, nan_columns)

def mask_nan_columns(batch: Batch, columns: list[str]) -> Batch:
    for column in columns:
        batch = mask_nan_column(batch, column)

    return batch

def mask_nan_column(batch: Batch, column: str) -> Batch:
    assert batch.metadata[column]['type'] == 'numerical', 'Can only mask NaNs in numerical columns. Use One-hot-Encoding for categorical columns.'
    target_column_index = batch.metadata[column]['index']

    masked_traces = []
    for trace in batch.traces:
        masked_trace = np.copy(trace)
        mask = np.isnan(trace[:, target_column_index].astype(float))
        masked_trace[mask, target_column_index] = 0.0
        masked_traces.append(np.concatenate((masked_trace, np.array([mask.astype(float)]).T), axis=1))

    new_metadata = deepcopy(batch.metadata)
    new_metadata[column]['masked'] = True
    new_metadata[column]['masked_column'] = len(masked_traces[0][0])-1

    return Batch(masked_traces, new_metadata)

def standardize_all_columns(batch: Batch) -> Batch:
    return standardize_columns(batch, [k for k,v in batch.metadata.items() if v['type']=='numerical'])

def standardize_columns(batch: Batch, columns: list[str]) -> Batch:
    for column in columns:
        batch = standardize_column(batch, column)

    return batch

def standardize_column(batch: Batch, column: str) -> Batch:
    assert batch.metadata[column]['type'] == 'numerical', 'Can only standardize numerical columns.'
    target_column_index = batch.metadata[column]['index']

    standardized_traces = []
    for trace in batch.traces:
        standardized_trace = np.copy(trace)
        standardized_trace[:, target_column_index] -= batch.metadata[column]['mean']
        standardized_trace[:, target_column_index] /= batch.metadata[column]['std']
        standardized_traces.append(standardized_trace)

    new_metadata = deepcopy(batch.metadata)
    new_metadata[column]['standardized'] = True

    return Batch(standardized_traces, new_metadata)


def pack_batch(batch: Batch, transform_timestamps=True, case_attribute_columns=False, initial_timestamp=True) -> PackedBatch:
    packed_traces = []
    packed_column_metadata = OrderedDict()
    num_case_attribute_columns = 0
    num_event_attribute_columns = 0
    num_timestamp_columns = 0
    values = {}
    for trace in batch.traces:
        packed_trace = []
        for position, (column, metadata) in enumerate(batch.metadata.items()):
            metadata_exists = column in packed_column_metadata

            if metadata['is_case_attribute'] and not case_attribute_columns:
                # is case attribute
                if metadata['type'] == 'numerical':
                    packed_trace.append(trace[0, metadata['index']])

                    if not metadata_exists:
                        num_case_attribute_columns += 1
                        packed_column_metadata[column] = {
                            'type': 'numerical',
                            'is_case_attribute': True,
                            'is_timestamp': False,
                            'unpacked_column_id': metadata['index'],
                            'standardized': metadata['standardized'],
                            'mean': metadata['mean'],
                            'std': metadata['std'],
                        }

                elif metadata['type'] == 'categorical':
                    packed_trace.append(trace[0, metadata['index']])

                    if not metadata_exists:
                        num_case_attribute_columns += 1
                        packed_column_metadata[column] = {
                            'type': 'categorical',
                            'is_case_attribute': True,
                            'unpacked_column_id': metadata['index'],
                            'unique_values': metadata['unique_values'],
                        }

                elif metadata['type'] == 'one_hot':
                    current_event = trace[0]
                    value_index = metadata['value_index']
                    unpacked_value_index = list(value_index.items())
                    one_hot_encoded_value = [current_event[index] for _, index in unpacked_value_index]
                    packed_trace.extend(one_hot_encoded_value)

                    if not metadata_exists:
                        num_case_attribute_columns += len(value_index)
                        packed_column_metadata[column] = {
                            'type': 'one_hot',
                            'is_case_attribute': True,
                            'unpacked_value_index': unpacked_value_index,
                        }

            else:
                # is event attribute
                if metadata['type'] == 'numerical':
                    if metadata['is_timestamp'] and transform_timestamps:
                        if initial_timestamp:
                            base_timestamp = trace[0, metadata['index']]
                            durations = np.ravel(trace[:, metadata['index']]) - \
                                np.ravel(np.concatenate(([base_timestamp], trace[:-1, metadata['index']])))

                            packed_trace.append(base_timestamp)
                            packed_trace.extend(durations)

                            if not metadata_exists:
                                num_timestamp_columns += 1

                            if column in values:
                                values[column]['durations'].extend(durations[1:])
                                values[column]['base'].append(base_timestamp)
                            else:
                                values[column] = {
                                    'durations': list(durations[1:]),
                                    'base': [base_timestamp]
                                }

                        else:
                            durations = np.ravel(trace[:, metadata['index']]) - \
                                np.ravel(np.concatenate(([0], trace[:-1, metadata['index']])))

                            packed_trace.extend(durations)

                            if column in values:
                                values[column]['durations'].extend(durations[1:])
                                values[column]['base'].append(durations[0])
                            else:
                                values[column] = {
                                    'durations': list(durations[1:]),
                                    'base': [durations[0]]
                                }

                    else:
                        packed_trace.extend(np.ravel(trace[:, metadata['index']]))

                    if not metadata_exists:
                        num_event_attribute_columns += 1
                        packed_column_metadata[column] = {
                            'type': 'numerical',
                            'is_case_attribute': metadata['is_case_attribute'],
                            'is_timestamp': metadata['is_timestamp'],
                            'unpacked_column_id': metadata['index'],
                            'standardized': metadata['standardized'],
                            'mean': metadata['mean'],
                            'std': metadata['std'],
                        }

                elif metadata['type'] == 'categorical':
                    packed_trace.extend(np.ravel(trace[:, metadata['index']]))

                    if not metadata_exists:
                        num_event_attribute_columns += 1
                        packed_column_metadata[column] = {
                            'type': 'categorical',
                            'is_case_attribute': metadata['is_case_attribute'],
                            'unpacked_column_id': metadata['index'],
                            'unique_values': metadata['unique_values'],
                        }

                elif metadata['type'] == 'one_hot':
                    # order: [e1oh1, e1oh2, e1oh3,....,e2oh1,e2oh2,...,enoh1,onoh2,...]
                    value_index = metadata['value_index']
                    unpacked_value_index = list(value_index.items())
                    for event in trace:
                        one_hot_encoded_value = [event[index] for _, index in unpacked_value_index]
                        packed_trace.extend(one_hot_encoded_value)

                    if not metadata_exists:
                        num_event_attribute_columns += len(value_index)
                        packed_column_metadata[column] = {
                            'type': 'one_hot',
                            'is_case_attribute': metadata['is_case_attribute'],
                            'unpacked_value_index': unpacked_value_index,
                        }

        packed_traces.append(np.array(packed_trace, dtype=object))

    for column, data in values.items():
        packed_column_metadata[column]['durations_mean'] = np.mean(data['durations'])
        packed_column_metadata[column]['durations_std'] = np.std(data['durations'])
        packed_column_metadata[column]['base_mean'] = np.mean(data['base'])
        packed_column_metadata[column]['base_std'] = np.std(data['base'])

    return PackedBatch(packed_traces, {'num_event_attribute_columns': num_event_attribute_columns,
                                       'num_case_attribute_columns': num_case_attribute_columns,
                                       'num_timestamp_columns': num_timestamp_columns,
                                       'transform_timestamps': transform_timestamps,
                                       'case_attribute_columns': case_attribute_columns,
                                       'initial_timestamp': initial_timestamp,
                                       'columns': packed_column_metadata})


def unpack_batch(packed_batch: PackedBatch) -> Batch:
    transform_timestamps = packed_batch.metadata['transform_timestamps']
    case_attribute_columns = packed_batch.metadata['case_attribute_columns']
    initial_timestamp = packed_batch.metadata['initial_timestamp']
    unpacked_traces = []
    unpacked_trace_metadata = {}
    for trace in packed_batch.traces:
        trace_length = (len(trace) -
                        packed_batch.metadata['num_timestamp_columns'] -
                        packed_batch.metadata['num_case_attribute_columns']) // \
                        packed_batch.metadata['num_event_attribute_columns']

        unpacked_trace = np.empty((trace_length, packed_batch.metadata['num_case_attribute_columns'] +
                                   packed_batch.metadata['num_event_attribute_columns']), dtype=object)
        cursor = 0
        for (column, metadata) in packed_batch.metadata['columns'].items():
            metadata_exists = column in unpacked_trace_metadata

            if metadata['is_case_attribute'] and not case_attribute_columns:
                # is case attribute
                if metadata['type'] == 'numerical':
                    unpacked_trace[:, metadata['unpacked_column_id']] = trace[cursor]

                    if not metadata_exists:
                        unpacked_trace_metadata[column] = {
                            'type': 'numerical',
                            'is_case_attribute': True,
                            'is_timestamp': metadata['is_timestamp'],
                            'index': metadata['unpacked_column_id'],
                            'standardized': metadata['standardized'],
                            'mean': metadata['mean'],
                            'std': metadata['std'],
                        }

                    cursor += 1

                elif metadata['type'] == 'categorical':
                    unpacked_trace[:, metadata['unpacked_column_id']] = trace[cursor]

                    if not metadata_exists:
                        unpacked_trace_metadata[column] = {
                            'type': 'categorical',
                            'is_case_attribute': True,
                            'index': metadata['unpacked_column_id'],
                            'unique_values': metadata['unique_values'],
                        }

                    cursor += 1

                elif metadata['type'] == 'one_hot':
                    for value, index in metadata['unpacked_value_index']:
                        unpacked_trace[:, index] = trace[cursor]
                        cursor += 1

                    if not metadata_exists:
                        unpacked_trace_metadata[column] = {
                            'type': 'one_hot',
                            'is_case_attribute': True,
                            'value_index': dict(metadata['unpacked_value_index']),
                        }

            else:
                # is event attribute
                if metadata['type'] == 'numerical':
                    if metadata['is_timestamp'] and transform_timestamps:
                        if initial_timestamp:
                            base_timestamp = trace[cursor + trace_length]
                        durations = trace[cursor:cursor + trace_length]
                        cumulative_durations = []
                        for duration in durations:
                            if not cumulative_durations:
                                if initial_timestamp:
                                    cumulative_durations.append(base_timestamp + duration)
                                else:
                                    cumulative_durations.append(duration)
                            else:
                                cumulative_durations.append(cumulative_durations[-1] + duration) 

                        unpacked_trace[:, metadata['unpacked_column_id']] = cumulative_durations
                        if initial_timestamp:
                            cursor += trace_length + 1
                        else:
                            cursor += trace_length

                    else:
                        unpacked_trace[:, metadata['unpacked_column_id']] = trace[cursor:cursor + trace_length]
                        cursor += trace_length

                    if not metadata_exists:
                        unpacked_trace_metadata[column] = {
                            'type': 'numerical',
                            'is_case_attribute': metadata['is_case_attribute'],
                            'is_timestamp': metadata['is_timestamp'],
                            'index': metadata['unpacked_column_id'],
                            'standardized': metadata['standardized'],
                            'mean': metadata['mean'],
                            'std': metadata['std'],
                        }

                elif metadata['type'] == 'categorical':
                    unpacked_trace[:, metadata['unpacked_column_id']] = trace[cursor:cursor + trace_length]

                    if not metadata_exists:
                        unpacked_trace_metadata[column] = {
                            'type': 'categorical',
                            'is_case_attribute': metadata['is_case_attribute'],
                            'index': metadata['unpacked_column_id'],
                            'unique_values': metadata['unique_values'],
                        }

                    cursor += trace_length

                elif metadata['type'] == 'one_hot':
                    num_values = len(metadata['unpacked_value_index'])
                    for value, index in metadata['unpacked_value_index']:
                        unpacked_trace[:, index] = [trace[cursor + num_values*i] for i in range(trace_length)]
                        cursor += 1

                    if not metadata_exists:
                        unpacked_trace_metadata[column] = {
                            'type': 'one_hot',
                            'is_case_attribute': metadata['is_case_attribute'],
                            'value_index': dict(metadata['unpacked_value_index']),
                        }

                    cursor += (trace_length - 1) * num_values

        unpacked_traces.append(np.array(unpacked_trace))

    return Batch(unpacked_traces, unpacked_trace_metadata)


class PackedBatch:
    @staticmethod
    def from_batch(batch: Batch, transform_timestamps=True, case_attribute_columns=False, initial_timestamp=True) -> PackedBatch:
        return pack_batch(batch, transform_timestamps, case_attribute_columns, initial_timestamp)

    def __init__(self, packed_traces: list[np.ndarray], metadata: dict):
        self.traces = packed_traces
        self.metadata = metadata

    @property
    def column_names(self) -> list[Literal[0, 1]]:
        assert len(np.unique([len(trace) for trace in self.traces])) == 1, 'To create a mask, the Batch may only contain traces of the same length.'
        column_names = []
        trace_length = (len(self.traces[0]) - self.metadata['num_case_attribute_columns']) // self.metadata['num_event_attribute_columns']
        for (column, metadata) in self.metadata['columns'].items():
            if metadata['is_case_attribute'] and not self.metadata['case_attribute_columns']:
                # is case attribute
                if metadata['type'] == 'numerical':
                    column_names.append(column)

                elif metadata['type'] == 'categorical':
                    column_names.append(column)

                elif metadata['type'] == 'one_hot':
                    for _ in metadata['unpacked_value_index']:
                        column_names.append(column)

            else:
                # is event attribute
                if metadata['type'] == 'numerical':
                    if metadata['is_timestamp'] and self.metadata['transform_timestamps'] and self.metadata['initial_timestamp']:
                        column_names.extend([column for _ in range(trace_length + 1)])
                    else:
                        column_names.extend([column for _ in range(trace_length)])

                elif metadata['type'] == 'categorical':
                    column_names.extend([column for _ in range(trace_length)])

                elif metadata['type'] == 'one_hot':
                    for _ in metadata['unpacked_value_index']:
                        column_names.extend([column for _ in range(trace_length)])

        return column_names

    def column_metadata_from_trace(self, trace) -> list[Dict]:
        column_metadata = []
        trace_length = (len(trace) - self.metadata['num_case_attribute_columns']) // self.metadata['num_event_attribute_columns']
        cursor = 0
        for (column, metadata) in self.metadata['columns'].items():
            if metadata['is_case_attribute'] and not self.metadata['case_attribute_columns']:
                # is case attribute
                if metadata['type'] == 'numerical':
                    column_metadata.append({
                        'column_name': column,
                        'type': 'numerical',
                        'is_case_attribute': True,
                        'is_timestamp': False,
                    })
                    cursor += 1

                elif metadata['type'] == 'categorical':
                    column_metadata.append({
                        'column_name': column,
                        'is_case_attribute': True,
                        'type': 'categorical',
                    })
                    cursor += 1

                elif metadata['type'] == 'one_hot':
                    for value, _ in metadata['unpacked_value_index']:
                        column_metadata.append({
                            'column_name': column,
                            'is_case_attribute': True,
                            'type': 'one-hot',
                            'value': value,
                        })
                        cursor += 1

            else:
                # is event attribute
                if metadata['type'] == 'numerical':
                    if metadata['is_timestamp'] and self.metadata['transform_timestamps'] and self.metadata['initial_timestamp']:
                        column_metadata.append({
                            'column_name': column,
                            'is_case_attribute': True,
                            'type': 'numerical',
                            'is_timestamp': True,
                            'timestamp_type': 'base'
                        })
                        cursor += 1
                        for event in range(trace_length):
                            column_metadata.append({
                                'column_name': column,
                                'event': event,
                                'is_case_attribute': False,
                                'type': 'numerical',
                                'is_timestamp': True,
                                'timestamp_type': 'durations'
                            })
                            cursor += 1
                    elif metadata['is_timestamp'] and self.metadata['transform_timestamps']:
                        column_metadata.append({
                            'column_name': column,
                            'event': 0,
                            'is_case_attribute': False,
                            'type': 'numerical',
                            'is_timestamp': True,
                            'timestamp_type': 'base'
                        })
                        cursor += 1
                        for event in range(1, trace_length):
                            column_metadata.append({
                                'column_name': column,
                                'event': event,
                                'is_case_attribute': False,
                                'type': 'numerical',
                                'is_timestamp': True,
                                'timestamp_type': 'durations'
                            })
                            cursor += 1
                    else:
                        for event in range(trace_length):
                            column_metadata.append({
                                'column_name': column,
                                'event': event,
                                'is_case_attribute': False,
                                'type': 'numerical',
                                'is_timestamp': False,
                            })
                            cursor += 1

                elif metadata['type'] == 'categorical':
                    for event in range(trace_length):
                        column_metadata.append({
                            'column_name': column,
                            'event': event,
                            'is_case_attribute': False,
                            'type': 'categorical',
                        })
                        cursor += 1

                elif metadata['type'] == 'one_hot':
                    for value, _ in metadata['unpacked_value_index']:
                        for event in range(trace_length):
                            column_metadata.append({
                                'column_name': column,
                                'event': event,
                                'is_case_attribute': False,
                                'type': 'one-hot',
                                'value': value,
                            })
                            cursor += 1

        return column_metadata

class Batch:
    @staticmethod
    def from_packed_batch(packed_batch: PackedBatch) -> Batch:
        return unpack_batch(packed_batch)

    def __init__(self, traces: list[np.ndarray], metadata: dict):
        self.metadata = metadata
        self.traces = traces
        self._validate_traces()

    def append_trace(self, trace: np.ndarray):
        self.append_traces([trace])

    def append_traces(self, traces: list[np.ndarray]):
        self.traces.extend(traces)
        self._validate_traces()

    def _validate_traces(self):
        feature_vector_length = sum([1 if info['type'] in ('categorical', 'numerical') else len(info['value_index'])
                                     for info in self.metadata.values()])
        for trace in self.traces:
            self._validate_trace(trace, feature_vector_length)

    @staticmethod
    def _validate_trace(trace, feature_vector_length):
        dimension_error = 'Trace has to be a 2-dimensional numpy array! Your array was %i-dimensional.'
        assert trace.ndim == 2, dimension_error % trace.ndim

        vector_error = 'The feature vector has to be of length %i! Your feature vectors are of length %i.'
        #assert trace.shape[1] == feature_vector_length, vector_error % (feature_vector_length, trace.shape[1])


class Dataset(ABC):

    @staticmethod
    def delimiter():
        raise NotImplementedError

    @staticmethod
    def categorical_columns():
        raise NotImplementedError

    @staticmethod
    def numerical_columns():
        raise NotImplementedError

    @staticmethod
    def case_attribute_columns():
        raise NotImplementedError

    @staticmethod
    def timestamp_columns():
        raise NotImplementedError

    @staticmethod
    def delete_columns():
        raise NotImplementedError

    @staticmethod
    def _get_trace_label(trace: pd.DataFrame) -> Literal[0, 1]:
        raise NotImplementedError

    @staticmethod
    def trace_column_name() -> str:
        raise NotImplementedError

    @classmethod
    def from_csv(cls, path_to_csv) -> Dataset:
        traces, labels, metadata = Dataset._load_dataset(cls, path_to_csv)
        dataset = Dataset(traces, labels, metadata)
        with open(path_to_csv.replace('.csv', '.pkl'), 'wb') as fh:
            pickle.dump(dataset, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return dataset

    @staticmethod
    def from_pickle(path_to_pickle) -> Dataset:
        with open(path_to_pickle, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def _split_into_traces(cls, dataset: pd.DataFrame) -> list[pd.DataFrame]:
        # split into individual traces
        traces = [v for _, v in dataset.groupby(cls.trace_column_name())]

        return traces

    @staticmethod
    def _delete_columns(cls, traces):
        # drop case id as it is not relevant for the classifier
        for trace in traces:
            for column in cls.delete_columns():
                trace.drop(column, inplace=True, axis=1)

        return traces

    @staticmethod
    def _delete_unfinished_traces(traces: list[pd.DataFrame], labels: list[Literal[0, 1]]):
        finished_traces = [trace for i, trace in enumerate(traces) if labels[i] is not None]
        finished_labels = [label for label in labels if label is not None]
        return finished_traces, finished_labels

    @staticmethod
    def _convert_pd_traces_to_np(traces: list[pd.DataFrame]) -> list[np.ndarray]:
        return [trace.to_numpy() for trace in traces]

    @staticmethod
    def _get_trace_labels(cls, traces: list[pd.DataFrame]) -> list[Literal[0, 1]]:
        return [cls._get_trace_label(trace) for trace in traces]

    @staticmethod
    def _extract_metadata(dataset, numerical_columns: list[str], categorical_columns: list[str],
                          case_attribute_columns: list[str], timestamp_columns: list[str],
                          delete_columns: list[str]) -> dict:
        metadata = {}
        columns_in_order = list(dataset.columns)
        for column in delete_columns:
            columns_in_order.remove(column)
        for numerical_column in numerical_columns:
            metadata[numerical_column] = {
                'type': 'numerical',
                'index': columns_in_order.index(numerical_column),
                'is_case_attribute': numerical_column in case_attribute_columns,
                'is_timestamp': numerical_column not in case_attribute_columns and numerical_column in timestamp_columns,
                'standardized': False,
                'mean': dataset[numerical_column].mean(),
                'std': dataset[numerical_column].std(),
            }

        for categorical_column in categorical_columns:
            metadata[categorical_column] = {
                'type': 'categorical',
                'index': columns_in_order.index(categorical_column),
                'is_case_attribute': categorical_column in case_attribute_columns,
                'unique_values': list(dataset[categorical_column].unique()),
            }

        return metadata

    @staticmethod
    def _load_dataset(cls, path_to_csv) -> (list, list):
        raw_dataset = pd.read_csv(path_to_csv, delimiter=cls.delimiter())
        metadata = Dataset._extract_metadata(
            raw_dataset,
            categorical_columns=cls.categorical_columns(),
            numerical_columns=cls.numerical_columns(),
            case_attribute_columns=cls.case_attribute_columns(),
            timestamp_columns=cls.timestamp_columns(),
            delete_columns=cls.delete_columns()
        )
        traces = Dataset._split_into_traces(cls, raw_dataset)
        labels = Dataset._get_trace_labels(cls, traces)
        traces, labels = Dataset._delete_unfinished_traces(traces, labels)
        traces = Dataset._delete_columns(cls, traces)
        return Dataset._convert_pd_traces_to_np(traces), labels, metadata

    @property
    def dataset(self) -> tuple[Batch, list]:
        return Batch(self._traces, self.metadata), self._labels

    @property
    def training_set(self) -> tuple[Batch, list]:
        number_of_traces = len(self._traces)
        training_traces_until = int(number_of_traces * 2/3)
        return Batch(self._traces[:training_traces_until], self.metadata), self._labels[:training_traces_until]

    @property
    def test_set(self) -> tuple[Batch, list]:
        number_of_traces = len(self._traces)
        training_traces_from = int(number_of_traces * 2/3)
        return Batch(self._traces[training_traces_from:], self.metadata), self._labels[training_traces_from:]

    @staticmethod
    def _get_element_by_value(haystack: dict, needle: any):
        for key, value in haystack.items():
            if value == needle:
                return key
        raise ValueError(f'{str(needle)} not a value in {str(haystack)}!')

    def get_column_name_by_index(self, index: int):
        for column_name, metadata in self.metadata.items():
            is_one_hot_column = metadata['type'] == 'one_hot' and index in metadata['value_index'].values()
            is_other_column = metadata['type'] != 'one_hot' and metadata['index'] == index

            if is_one_hot_column or is_other_column:
                return column_name
        raise ValueError('Index out of bounds.')

    def get_one_hot_value_by_index(self, index: str):
        for column_name, metadata in self.metadata.items():
            if metadata['type'] != 'one_hot':
                continue

            try:
                return Dataset._get_element_by_value(metadata['value_index'], index)
            except ValueError:
                pass

        raise ValueError('Index out of bounds or not a one hot encoded column.')

    def generator(self, batch_size) -> Generator:
        while True:
            permutation = np.random.permutation(len(self.dataset[1]))
            for i in range(0, len(self.dataset[1]), batch_size):
                try:
                    max_size = min(batch_size, len(self.dataset[1]) - i)
                    traces = [self.dataset[0].traces[permutation[i + j]] for j in range(max_size)]
                    labels = [self.dataset[1][permutation[i + j]] for j in range(max_size)]
                    new_dataset = Dataset(traces, labels, self.metadata)
                    yield new_dataset
                except IndexError:
                    yield None

    def __init__(self, traces, labels, metadata):
        self._traces = traces
        self._labels = labels
        self.metadata = metadata
