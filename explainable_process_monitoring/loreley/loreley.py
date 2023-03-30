from __future__ import annotations

import collections
import math
import random
from enum import Enum
from typing import Literal

import numpy as np
from predictive_process_monitoring.prbpm_models.dataset import Dataset, standardize_columns, Batch, one_hot_encode_all_columns, pack_batch, \
    unpack_batch, PackedBatch
from predictive_process_monitoring.prbpm_models.interface import BinaryClassificationModelInterface
from sklearn import tree
from sklearn.neighbors import DistanceMetric, KernelDensity

from loreley.utils import nan_safe_equals, Logger


class DatasetConfig:

    def __init__(self,
                 metadata=None,
                 ga_target_size=600,
                 ga_number_of_generations=15,
                 crossover_chance=0.7,
                 mutation_chance=0.15,
                 use_packing=True,
                 event_type_feature_index=0,
                 transform_timestamps=True,
                 case_attribute_columns=False,
                 initial_timestamp=True,
                 ):
        self.metadata: dict = metadata  # dataset attribute metadata informing us about event type / categorical / numerical attributes etc.
        self.ga_target_size = ga_target_size
        self.ga_number_of_generations = ga_number_of_generations
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        self.oh_meta: dict = {}
        self.transform_timestamps = transform_timestamps
        self.case_attribute_columns = case_attribute_columns
        self.initial_timestamp = initial_timestamp
        self.event_type_feature_index=event_type_feature_index
        self.use_packing = use_packing


class Loreley:

    def __init__(self, blackbox: BinaryClassificationModelInterface, trace_dataset: Dataset,
                 dataset_config: DatasetConfig, print_debug: bool = False):
        self.integer_encoded_values = list()
        self.blackbox = blackbox
        self.dataset_config = dataset_config
        self.trace_dataset = trace_dataset
        self.replace_nan_in_numeric(
            self.trace_dataset.test_set[0].traces)  # replace NaN values with a numerical replacement value
        self.integer_encode_categorical_strings(self.trace_dataset.test_set[
                                                    0].traces)  # integer encode categorical strings ( as some functions can't handle strings)
        self._crossover_chance = dataset_config.crossover_chance
        self._mutation_chance = dataset_config.mutation_chance
        self.construct_value_distributions()  # kernel distribution for mutation
        self.print_debug = print_debug

    def replace_nan_in_numeric(self, traces: np.ndarray):
        for meta in self.dataset_config.metadata.values():
            if meta['type'] == 'numerical':
                for trace in traces:
                    for event in trace:
                        event[meta['index']] = np.nan_to_num(event[meta['index']], nan=-1)

    def construct_value_distributions(self):
        # get all events into one list
        events = [event for trace in self.trace_dataset.test_set[0].traces for event in np.split(trace, len(trace))]
        # transform the list of traces to a list of events of all traces.
        events = np.array(events).reshape(-1, len(self.dataset_config.metadata))
        events = np.nan_to_num(events, nan=-1)
        self._kde = KernelDensity(bandwidth=0.01, kernel='gaussian').fit(
            np.nan_to_num([[x for x in e] for e in events], nan=-1))

    def integer_encode_categorical_strings(self, traces: np.ndarray):
        for meta in self.dataset_config.metadata.values():
            if meta['type'] == 'categorical' and type(meta['unique_values'][0]) == str:
                self.integer_encoded_values.append(meta['index'])
                u_values: list = meta['unique_values']
                for t in traces:
                    for e in t:
                        i = [idx for idx, x in enumerate(u_values) if x == e[meta['index']]][0]
                        e[meta['index']] = i

    def string_decode_categorical_integers(self, traces: np.ndarray):
        for meta in self.dataset_config.metadata.values():
            if meta['type'] == 'categorical' and type(meta['unique_values'][0]) == str:
                u_values: list = meta['unique_values']
                for t in traces:
                    for e in t:
                        e[meta['index']] = u_values[e[meta['index']]]

    def get_initial_population(self, trace: np.ndarray, traces: list[np.ndarray]):
        same_length_traces = self.cut_traces_to_same_length(len(trace), traces)
        initial_population = [
            t for t in same_length_traces if self.calculate_edit_distance(trace, t) == 0]
        initial_population = self.expand_population_to_target_size(initial_population)
        return initial_population

    def expand_population_to_target_size(self, population: list[np.ndarray]) -> list[np.ndarray]:
        target_size = self.dataset_config.ga_target_size
        while len(population) < target_size:
            population = population + [np.copy(p) for p in population]
        return population[:target_size]

    def cut_traces_to_same_length(self, length: int, traces: list[np.ndarray]):
        return [t[:length, :] for t in traces if
                len(t) >= length]  # takes all traces and cuts them down to a fixed prefix length

    def get_traces_with_same_length(self, trace_to_compare_to: np.ndarray, traces: list[np.ndarray]):
        desired_len = len(trace_to_compare_to)
        traces_with_same_length = [x for x in traces if len(x) == desired_len]
        return traces_with_same_length

    def calculate_edit_distance(self, trace_1: np.ndarray, trace_2: np.ndarray):
        t1_event_type_attribute = trace_1[:, self.dataset_config.event_type_feature_index]
        t2_event_type_attribute = trace_2[:, self.dataset_config.event_type_feature_index]
        flat_t_1 = t1_event_type_attribute.flatten()
        flat_t_2 = t2_event_type_attribute.flatten()
        if (flat_t_1 == flat_t_2).all():
            return 0
        distances = np.zeros((len(flat_t_1) + 1, len(flat_t_2) + 1))
        for t1 in range(len(flat_t_1) + 1):
            distances[t1][0] = t1
        for t2 in range(len(flat_t_2) + 1):
            distances[0][t2] = t2

        for t1 in range(1, len(flat_t_1) + 1):
            for t2 in range(1, len(flat_t_2) + 1):
                if (flat_t_1[t1 - 1] == flat_t_2[t2 - 1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    if (a <= b and a <= c):
                        distances[t1][t2] = a + 1
                    elif (b <= a and b <= c):
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1
        return distances[len(flat_t_1)][len(flat_t_2)]

    def crossover(self, current_gen: list[np.ndarray], metadata: dict, num_events: int):
        crossover_chance = self._crossover_chance
        random_values = np.random.uniform(size=len(current_gen))
        should_crossover = random_values <= crossover_chance
        new_traces = []
        for i in range(len(current_gen)):
            if should_crossover[i]:
                if self.dataset_config.use_packing:
                    new_traces.append(self.do_crossover_packed(current_gen[i], current_gen, metadata, num_events))
                else:
                    new_traces.append(self.do_crossover(current_gen[i], current_gen, metadata, num_events))
        return current_gen + new_traces

    def _number_of_attribute_entries(self, attribute_idx: int, metadata: dict, num_events: int):
        metadata_entry = metadata[attribute_idx]
        if metadata_entry['is_case_attribute']:
            return 1
        elif metadata_entry['is_timestamp'] and metadata['initial_timestamp']:
            return num_events + 1
        else:
            return num_events

    def _get_next_border(self, current_idx, attribute_idx, metadata, num_events):
        return current_idx + self._number_of_attribute_entries(attribute_idx, metadata, num_events)

    def do_crossover_packed(self, trace: np.ndarray, current_gen: list[np.ndarray], metadata: dict, num_events: int):
        new_trace = np.array(trace, copy=True)
        crossover_p_1 = np.random.randint(0, len(new_trace) - 1)
        crossover_p_2 = np.random.randint(crossover_p_1, len(new_trace))
        random_trace = current_gen[np.random.randint(0, len(current_gen))]
        new_trace[crossover_p_1:crossover_p_2] = random_trace[crossover_p_1:crossover_p_2]
        return new_trace

    def do_crossover(self, trace: np.ndarray, current_gen: list[np.ndarray], metadata: dict,
                     num_events: int) -> np.ndarray:
        event_type_attributes = []
        new_trace = np.array(trace, copy=True)
        number_of_values = len(new_trace) * len(new_trace[0])
        crossover_start = np.random.randint(0, number_of_values)
        crossover_end = np.random.randint(crossover_start, number_of_values)
        random_trace_index = np.random.randint(0, len(current_gen))
        for attribute_idx in range(len(new_trace[0])):
            if [*metadata.values()][attribute_idx]['is_event_id']:
                event_type_attributes.append(attribute_idx)
            else:
                # do crossover with random value

                for i in range(len(new_trace)):
                    current_idx = attribute_idx * len(new_trace) + i
                    if crossover_start <= current_idx < crossover_end:
                        new_trace[i, attribute_idx] = current_gen[random_trace_index][i, attribute_idx]
        # for all event type attribute do crossover with one chosen trace
        for attribute_idx in range(len(event_type_attributes)):
            current_idx = attribute_idx * len(new_trace)
            if crossover_start <= current_idx < crossover_end:
                for i in range(len(new_trace)):
                    new_trace[i, attribute_idx] = current_gen[random_trace_index][i, attribute_idx]
        return new_trace

    def mutate(self, current_gen: list[np.ndarray], initial_gen: list[np.ndarray], metadata):
        if self.dataset_config.use_packing:
            _current_gen = unpack_batch(PackedBatch(current_gen, metadata)).traces
            metadata = self.dataset_config.metadata
        else:
            _current_gen = current_gen
        for trace in _current_gen:
            categorical_attributes = []
            mutation_trace = self._kde.sample(len(trace))  # create random trace
            for attribute_idx in range(len(trace[0])):
                if [x['type'] for x in [*metadata.values()] if x['index'] == attribute_idx][0] == 'categorical':
                    # not numeric
                    categorical_attributes.append(attribute_idx)
                else:
                    # do mutation
                    for i in range(len(trace)):
                        should_mutate = np.random.uniform() <= self._mutation_chance
                        if should_mutate:
                            trace[i, attribute_idx] = mutation_trace[i, attribute_idx]
            should_mutate_event = np.random.uniform() <= self._mutation_chance
            if should_mutate_event:  # TODO currently only prefix mutation
                random_sample = initial_gen[np.random.randint(0, len(initial_gen))]
                for attribute_idx in categorical_attributes:
                    if [x['is_event_id'] for x in [*metadata.values()] if x['index'] == attribute_idx][0]:
                        for i in range(len(trace)):
                            trace[i, attribute_idx] = random_sample[i, attribute_idx]
                    else:
                        metadata_val = [x for x in [*metadata.values()] if x['index'] == attribute_idx][0]
                        possible_vals = metadata_val['unique_values']
                        is_case_attribute = metadata_val['is_case_attribute']
                        possible_vals_integer_encoded = [i for i in range(len(possible_vals))]
                        is_integer_encoded = attribute_idx in self.integer_encoded_values
                        if is_case_attribute and self.dataset_config.use_packing and not self.dataset_config.case_attribute_columns:
                            new_val = np.random.choice(
                                possible_vals_integer_encoded if is_integer_encoded else possible_vals)
                            for i in range(len(trace)):
                                trace[i, attribute_idx] = new_val
                        else:
                            for i in range(len(trace)):
                                new_val = np.random.choice(
                                    possible_vals_integer_encoded if is_integer_encoded else possible_vals)
                                trace[i, attribute_idx] = new_val

        return _current_gen

    def select(self, current_gen: list[np.ndarray], trace_x: np.ndarray, target_prediction: any):

        fitness_scores = self.calculate_fitness(current_gen, trace_x, target_prediction)
        min_fitness = sorted(fitness_scores, reverse=True)[self.dataset_config.ga_target_size - 1]
        selected_traces = [trace for index, trace in enumerate(current_gen) if fitness_scores[index] >= min_fitness]
        return selected_traces[:self.dataset_config.ga_target_size]

    def calculate_fitness(self, traces: list[np.ndarray], trace_x: np.ndarray, target_prediction: any) -> list[float]:
        numerical_attribute_keys = [attr_name for attr_name, v in
                                    self.dataset_config.metadata.items() if v['type'] == 'numerical']
        standardized_traces = standardize_columns(Batch(traces, self.blackbox.get_dataset().metadata),
                                                  numerical_attribute_keys).traces
        standardized_trace_x = \
            standardize_columns(Batch([trace_x], self.blackbox.get_dataset().metadata),
                                numerical_attribute_keys).traces[0]
        distances = np.array(self.calculate_distance(standardized_traces, standardized_trace_x))
        distances *= 1 / np.max(distances)  # normalize distances
        blackbox_prediction_indicator = np.array(
            self.calculate_blackbox_prediction_indicator(traces, target_prediction))
        is_same_trace_indicator = np.array(self.calculate_is_same_trace_indicator(traces, trace_x))
        return blackbox_prediction_indicator + (1 - distances) - is_same_trace_indicator

    def calculate_blackbox_prediction_indicator(self, traces: list[np.ndarray], target_prediction: any) -> list[
        Literal[0, 1]]:
        t_cp = [t.copy() for t in traces]
        self.string_decode_categorical_integers(t_cp)
        predictions = self.blackbox.predict_binary_class_for_traces(t_cp)
        is_same_class = [1 if prediction == target_prediction else 0 for prediction in predictions]
        return is_same_class

    def calculate_is_same_trace_indicator(self, traces: list[np.ndarray], trace_x: np.ndarray) -> list[float]:
        return [1 if nan_safe_equals(trace, trace_x) else 0 for trace in traces]

    def calculate_distance(self, traces: list[np.ndarray], trace_x: np.ndarray) -> list[float]:
        jaccard_dist = DistanceMetric.get_metric("jaccard")
        euclidian_dist = DistanceMetric.get_metric("euclidean")
        distances = []
        for trace in traces:
            attr_distances = []
            for col_idx, column in enumerate(trace[0]):
                is_numeric = [x for x in [*self.dataset_config.metadata.values()]
                              if x['index'] == col_idx][0]['type'] == 'numerical'
                is_case_attribute = [x for x in [*self.dataset_config.metadata.values()]
                                     if x['index'] == col_idx][0]['is_case_attribute']
                distance = None
                trace_x_attr_vector = trace_x[:, col_idx]
                trace_attr_vector = trace[:, col_idx]
                if is_numeric:
                    # trace_x_attr_vector = [np.nan_to_num(x, nan=-1) for x in trace_x_attr_vector]
                    # trace_attr_vector = [np.nan_to_num(x, nan=-1) for x in trace_attr_vector]
                    distance = euclidian_dist.pairwise([trace_x_attr_vector, trace_attr_vector])[0][1]
                else:

                    if is_case_attribute and not self.dataset_config.case_attribute_columns:
                        distance = jaccard_dist.pairwise([[trace_x_attr_vector[0]], [trace_attr_vector[0]]])[0][1]
                    else:
                        distance = jaccard_dist.pairwise([trace_x_attr_vector, trace_attr_vector])[0][1]
                attr_distances.append(distance)
            distances.append(np.sum(attr_distances))
        return distances

    def run_genetic_algorithm(self, initial_population: any, trace: any, target_prediction: any):
        initial_gen_batch = Batch([x.copy() for x in initial_population], self.dataset_config.metadata)
        use_packing = self.dataset_config.use_packing
        num_events = len(initial_population[0])
        current_gen = initial_population
        for i in range(0, self.dataset_config.ga_number_of_generations):
            if use_packing:
                current_gen = pack_batch(Batch(current_gen, self.dataset_config.metadata),
                                         initial_timestamp=self.dataset_config.initial_timestamp,
                                         transform_timestamps=self.dataset_config.transform_timestamps,
                                         case_attribute_columns=self.dataset_config.case_attribute_columns)
                metadata = current_gen.metadata
                current_gen = current_gen.traces
            else:
                metadata = self.dataset_config.metadata
            current_gen = self.crossover(current_gen, metadata, num_events)
            if self.print_debug:
                print('After crossover')
                self.count_unique_traces(current_gen)
            current_gen = self.mutate(current_gen, initial_population, metadata)
            if self.print_debug:
                print('After mutation')
                self.count_unique_traces(current_gen)
            current_gen = self.select(current_gen, trace, target_prediction)
            if self.print_debug:
                print('After selection')
                self.count_unique_traces(current_gen)
        return current_gen[:self.dataset_config.ga_target_size]  # can be a bit more than target size, remove that

    def train_interpretable_model_packed(self, synthetic_traces: np.ndarray) -> \
            tuple[tree.DecisionTreeClassifier, float]:

        copy_s = [x.copy() for x in synthetic_traces]
        self.string_decode_categorical_integers(copy_s)

        oh_encoded_traces = one_hot_encode_all_columns(Batch(copy_s, metadata=self.dataset_config.metadata))
        self.oh_meta = oh_encoded_traces.metadata

        packed_batch = pack_batch(oh_encoded_traces, case_attribute_columns=self.dataset_config.case_attribute_columns,
                                  initial_timestamp=self.dataset_config.initial_timestamp,
                                  transform_timestamps=self.dataset_config.transform_timestamps)
        t_cp = [x.copy() for x in synthetic_traces]
        self.string_decode_categorical_integers(t_cp)
        predictions = self.blackbox.predict_binary_class_for_traces(t_cp)
        seed = random.randint(0, 1000)
        shuffled_oh_traces = packed_batch.traces.copy()
        random.Random(seed).shuffle(shuffled_oh_traces)
        shuffled_predictions = predictions.copy()
        random.Random(seed).shuffle(shuffled_predictions)
        shuffled_traces = synthetic_traces.copy()
        random.Random(seed).shuffle(shuffled_traces)
        train_test_split_index = int(self.dataset_config.ga_target_size * 2 * 0.8)
        classifier = tree.DecisionTreeClassifier(max_depth=5, min_impurity_decrease=0.005)
        classifier.fit(shuffled_oh_traces[:train_test_split_index], shuffled_predictions[:train_test_split_index])
        t_cp = [x.copy() for x in shuffled_traces[train_test_split_index:]]
        self.string_decode_categorical_integers(t_cp)
        counts = collections.Counter(
            classifier.predict(
                shuffled_oh_traces[train_test_split_index:]) == self.blackbox.predict_binary_class_for_traces(
                t_cp))

        return classifier, counts[True] / len(shuffled_oh_traces[train_test_split_index:])

    def train_interpretable_model(self, synthetic_traces: np.ndarray) -> tuple[
        tree.DecisionTreeClassifier, float]:

        """
        Trains an interpretable decision tree given a np array of bb encoded traces
        """

        if self.dataset_config.use_packing:
            return self.train_interpretable_model_packed(synthetic_traces)

        copy_s = [x.copy() for x in synthetic_traces]
        self.string_decode_categorical_integers(copy_s)

        oh_encoded_traces = one_hot_encode_all_columns(Batch(copy_s, metadata=self.dataset_config.metadata))
        self.oh_meta = oh_encoded_traces.metadata
        oh_encoded_traces = [np.array(trace).flatten() for trace in oh_encoded_traces.traces]

        t_cp = [x.copy() for x in synthetic_traces]
        self.string_decode_categorical_integers(t_cp)
        predictions = self.blackbox.predict_binary_class_for_traces(t_cp)
        seed = random.randint(0, 1000)
        shuffled_oh_traces = oh_encoded_traces.copy()
        random.Random(seed).shuffle(shuffled_oh_traces)
        shuffled_predictions = predictions.copy()
        random.Random(seed).shuffle(shuffled_predictions)
        shuffled_traces = synthetic_traces.copy()
        random.Random(seed).shuffle(shuffled_traces)
        train_test_split_index = int(self.dataset_config.ga_target_size * 2 * 0.8)
        classifier = tree.DecisionTreeClassifier(max_depth=5, min_impurity_decrease=0.005)
        classifier.fit(shuffled_oh_traces[:train_test_split_index], shuffled_predictions[:train_test_split_index])
        t_cp = [x.copy() for x in shuffled_traces[train_test_split_index:]]
        self.string_decode_categorical_integers(t_cp)
        counts = collections.Counter(
            classifier.predict(
                shuffled_oh_traces[train_test_split_index:]) == self.blackbox.predict_binary_class_for_traces(
                t_cp))

        return classifier, counts[True] / len(shuffled_oh_traces[train_test_split_index:])

    def generate_explanation_packed(self, clf: tree.DecisionTreeClassifier, pred_trace: np.ndarray) -> Explanation:
        copy_s = [x.copy() for x in pred_trace]
        self.string_decode_categorical_integers([copy_s])

        oh_v = one_hot_encode_all_columns(Batch([np.array(copy_s)], self.dataset_config.metadata))
        packed_v = pack_batch(oh_v, case_attribute_columns=self.dataset_config.case_attribute_columns,
                              transform_timestamps=self.dataset_config.transform_timestamps,
                              initial_timestamp=self.dataset_config.initial_timestamp)

        node_indicator = clf.decision_path(packed_v.traces)
        leaf_id = clf.apply(packed_v.traces)

        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        impurity = clf.tree_.impurity

        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]

        explanation_terms = []
        metadata = packed_v.column_metadata_from_trace(packed_v.traces[0])

        for node_id in node_index:
            if leaf_id[0] == node_id:
                continue
            # check if value of the split feature for sample 0 is below threshold
            if packed_v.traces[0][feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            feature_name = metadata[feature[node_id]]['column_name']
            feature_index = self.dataset_config.metadata[feature_name]['index']
            if metadata[feature[node_id]]['is_case_attribute']:
                event_idx = 0
                is_case_attribute = True
            else:
                if 'is_timestamp' in metadata[feature[node_id]] and metadata[feature[node_id]]['is_timestamp'] and \
                        metadata[feature[node_id]]['timestamp_type'] == 'base':
                    event_idx = 0
                else:
                    event_idx = metadata[feature[node_id]]['event']
                is_case_attribute = False

            imp = impurity[node_id]

            if metadata[feature[node_id]]['type'] == 'one-hot':
                feature_value = metadata[feature[node_id]]['value']
                if metadata[feature[node_id]]['value'] == copy_s[0][
                    self.dataset_config.metadata[feature_name]['index']]:
                    equality = '='
                else:
                    equality = '!='
                term_type = ExplanationTermType.CATEGORICAL
            else:
                if 'is_timestamp' in metadata[feature[node_id]] and metadata[feature[node_id]]['is_timestamp'] and \
                        metadata[feature[node_id]]['timestamp_type'] == 'durations':
                    base = [i for i, x in enumerate(metadata) if
                            x['column_name'] == feature_name and x['timestamp_type'] == 'base'][0]
                    threshold = base + threshold
                feature_value = pred_trace[event_idx][feature_index]
                equality = ''
                term_type = ExplanationTermType.NUMERICAL

            term = ExplanationTerm(event_idx=event_idx, feature_label=feature_name, feature_value=feature_value,
                                   inequality=threshold_sign, threshold=threshold[node_id], equality=equality,
                                   term_type=term_type, impurity=imp, feature_index=feature_index,
                                   actual_value=copy_s[event_idx][feature_index], is_case_attribute=is_case_attribute)

            explanation_terms.append(term)

        feature_importances = []

        for imp_tupl in [(i, x) for i, x in enumerate(clf.feature_importances_) if x != 0]:
            feature_name = metadata[imp_tupl[0]]['column_name']
            feature_index = self.dataset_config.metadata[feature_name]['index']
            if metadata[feature[node_id]]['is_case_attribute']:
                event_idx = 0
            elif 'is_timestamp' in metadata[feature[node_id]] and metadata[feature[node_id]]['is_timestamp'] and \
                    metadata[feature[node_id]]['timestamp_type'] == 'base':
                event_idx = 0
            else:
                event_idx = metadata[feature[node_id]]['event']
            feature_importances.append(
                FeatureImportance(event=event_idx, feature=feature_index, importance=imp_tupl[1],
                                  is_case_attribute=metadata[feature[node_id]]['is_case_attribute']))

        return Explanation(explanation_trace=pred_trace, prediction=clf.predict(packed_v.traces)[0],
                           explanation_terms=explanation_terms,
                           feature_importances=feature_importances)

    def generate_explanation(self, clf: tree.DecisionTreeClassifier, pred_trace: np.ndarray) -> Explanation:
        """
        prints an explanation for a trace given a classifier
        """
        copy_s = [x.copy() for x in pred_trace]
        self.string_decode_categorical_integers([copy_s])

        oh_v = one_hot_encode_all_columns(Batch([np.array(copy_s)], self.dataset_config.metadata))
        v = [np.array(oh_v.traces[0]).flatten()]
        self.oh_meta = oh_v.metadata

        mod_v = np.shape(oh_v.traces[0])[1]

        node_indicator = clf.decision_path(v)
        leaf_id = clf.apply(v)

        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        impurity = clf.tree_.impurity

        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]

        explanation_terms = []

        for node_id in node_index:
            if leaf_id[0] == node_id:
                continue
            # check if value of the split feature for sample 0 is below threshold
            if v[0][feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            imp = impurity[node_id]

            feature_idx = feature[node_id] % mod_v
            event_idx = math.floor(feature[node_id] / mod_v)
            bb_feature_idx = 0
            categorical_label = ''
            for key, value in self.oh_meta.items():
                if value['type'] == 'numerical':
                    start = value['index']
                    end = value['index']
                elif value['type'] == 'one_hot':
                    start = min(value['value_index'].values())
                    end = max(value['value_index'].values()) + 1
                else:
                    raise Exception('This should not happen. One Hot encoded data should only contain numerical or '
                                    'one_hot data')

                feature_label = key
                if start == end and start == feature_idx:
                    # numeric
                    bb_feature_idx = value['index']
                    break
                elif start <= feature_idx < end:
                    categorical_label = [x for x in value['value_index'].items() if x[1] == feature_idx][0][0]
                    break

            if categorical_label != '':
                value = categorical_label
            else:
                value = v[0][feature[node_id]]

            if categorical_label != '':
                tr = ''
                ineq = ''
                if v[0][feature[node_id]] == 1:
                    equality = '='
                else:
                    equality = "!="
            else:
                tr = threshold[node_id]
                ineq = threshold_sign
                equality = '='

            term_type = ExplanationTermType.NUMERICAL if categorical_label == '' else ExplanationTermType.CATEGORICAL

            actual_value = copy_s[event_idx][
                [y['index'] for x, y in self.dataset_config.metadata.items() if x == feature_label][0]]

            explanation_terms.append(
                ExplanationTerm(event_idx, feature_label, value, ineq, tr, equality, term_type, imp, bb_feature_idx,
                                actual_value))

        # Extract feature importance
        importances = [(i, x) for i, x in enumerate(clf.feature_importances_) if x != 0]
        imp_event = [math.floor(x[0] / mod_v) for x in importances]
        imp_feature = [x[0] % mod_v for x in importances]
        imp_feature_bb = list()
        for feature_idx in imp_feature:
            for label, m in self.oh_meta.items():
                if m['type'] == 'one_hot':
                    start = min(m['value_index'].values())
                    end = max(m['value_index'].values())
                elif m['type'] == 'numerical':
                    start = m['index']
                    end = m['index']
                else:
                    raise Exception('This should not happen')
                if start == feature_idx or start <= feature_idx <= end:
                    imp_feature_bb.append(self.dataset_config.metadata[label]['index'])

        imp_list: list[FeatureImportance] = list()
        for i, im in enumerate(importances):
            imp_list.append(FeatureImportance(imp_event[i], imp_feature_bb[i], im[1]))

        return Explanation(pred_trace, clf.predict(v)[0], explanation_terms, imp_list)

    def get_explanation_for_trace(self, trace: any, cutoff: int) -> tuple[Explanation, float, bool]:
        prefix = np.copy(trace)[:(-1) * cutoff]
        initial_population = self.get_initial_population(prefix, self.trace_dataset.test_set[0].traces)
        target_range = [0, 1]
        synthetic_traces = []
        for target in target_range:
            synthetic_traces.extend(self.run_genetic_algorithm(
                initial_population, prefix, target))
        interpretable_model, rate_for_synth = self.train_interpretable_model(
            synthetic_traces)
        same = self.is_local_pred_same_as_black_box(interpretable_model, prefix)
        if self.dataset_config.use_packing:
            explanation: Explanation = self.generate_explanation_packed(interpretable_model, prefix)
        else:
            explanation: Explanation = self.generate_explanation(interpretable_model, prefix)
        return explanation, rate_for_synth, same

    def is_local_pred_same_as_black_box(self, clf: tree.DecisionTreeClassifier, trace: np.ndarray) -> bool:
        t_cp = trace.copy()
        self.string_decode_categorical_integers([t_cp])
        bb_result = self.blackbox.predict_binary_class_for_traces([t_cp])
        oh_trace = one_hot_encode_all_columns(Batch([t_cp], self.dataset_config.metadata))
        if self.dataset_config.use_packing:
            oh_trace = pack_batch(oh_trace, transform_timestamps=self.dataset_config.transform_timestamps,
                                  initial_timestamp=self.dataset_config.initial_timestamp,
                                  case_attribute_columns=self.dataset_config.case_attribute_columns)
        else:
            oh_trace.traces[0] = oh_trace.traces[0].flatten()
        local_result = clf.predict([oh_trace.traces[0]])
        return bool((bb_result == local_result)[0])

    def count_unique_traces(self, traces: list[np.ndarray]):

        known_traces = list()
        known_traces_count = list()

        for trace in traces:

            was_found = False

            for i, t in enumerate(known_traces):
                if (np.nan_to_num(t) == np.nan_to_num(trace)).all():
                    was_found = True
                    known_traces_count[i] += 1
                    break

            if not was_found:
                known_traces.append(trace)
                known_traces_count.append(1)

        print("Found " + str(len(known_traces)) + " distinct traces. With the distribution " + str(known_traces_count))


class Explanation:
    def __init__(self, explanation_trace: np.ndarray, prediction: any, explanation_terms: list[ExplanationTerm],
                 feature_importances: list[FeatureImportance]):
        self.explanation_trace = explanation_trace
        self.prediction = prediction
        self.explanation_terms = explanation_terms
        self.feature_importances = feature_importances

    def log_explanation(self, logger: Logger):
        violation = 'violation'
        if self.prediction == 0:
            violation = 'non-violation'
        logger.log_in_file("v \\rightarrow " + violation + " \\\\")
        if len(self.explanation_terms) == 0:
            pass
        for term in self.explanation_terms:
            if term.term_type == ExplanationTermType.NUMERICAL:
                ie = '>'
                if term.inequality == '<=':
                    ie = '\\leq'
                logger.log_in_file("(v[{eventIdx}, {feature}] = {value}) & "
                                   "{inequality} {threshold} \\\\".format(
                    feature=term.feature_label,
                    eventIdx=term.event_idx,
                    value=term.feature_value,
                    actual_value=term.actual_value,
                    inequality=ie,
                    threshold=term.threshold))
            else:
                e = '='
                if term.equality == '!=':
                    e = '\\neq'
                logger.log_in_file("(v[{eventIdx}, {feature}] = {actual_value}) & {equality} {value} \\\\".format(
                    equality=e,
                    feature=term.feature_label,
                    eventIdx=term.event_idx,
                    value=term.feature_value,
                    inequality=term.inequality,
                    threshold=term.threshold,
                    actual_value=term.actual_value))

    def print_explanation(self):
        violation = 'violation'
        if self.prediction == 0:
            violation = 'non-violation'
        print("v \\rightarrow " + violation + " \\\\")
        if len(self.explanation_terms) == 0:
            pass
        for term in self.explanation_terms:
            if term.term_type == ExplanationTermType.NUMERICAL:
                ie = '>'
                if term.inequality == '<=':
                    ie = '\\leq'
                print("(v[{eventIdx}, {feature}] = {value}) & "
                      "{inequality} {threshold} \\\\".format(
                    feature=term.feature_label,
                    eventIdx=term.event_idx,
                    value=term.feature_value,
                    actual_value=term.actual_value,
                    inequality=ie,
                    threshold=term.threshold))
            else:
                e = '='
                if term.equality == '!=':
                    e = '\\neq'
                print("(v[{eventIdx}, {feature}] = {actual_value}) & {equality} {value} \\\\".format(
                    equality=e,
                    feature=term.feature_label,
                    eventIdx=term.event_idx,
                    value=term.feature_value,
                    inequality=term.inequality,
                    threshold=term.threshold,
                    actual_value=term.actual_value))


class ExplanationTermType(Enum):
    NUMERICAL = 1
    CATEGORICAL = 2


class ExplanationTerm:
    def __init__(self, event_idx: int, feature_label: str, feature_value: str, inequality: str, threshold: str,
                 equality: str, term_type: ExplanationTermType, impurity: float, feature_index: int, actual_value: any,
                 is_case_attribute: bool = False):
        self.event_idx = event_idx
        self.feature_label = feature_label
        self.feature_value = feature_value
        self.inequality = inequality
        self.threshold = threshold
        self.equality = equality
        self.term_type = term_type
        self.impurity = impurity
        self.feature_index = feature_index
        self.actual_value = actual_value
        self.is_case_attribute = is_case_attribute


class FeatureImportance:
    def __init__(self, event: int, feature: int, importance: float, is_case_attribute: bool = False):
        self.event = event
        self.feature = feature
        self.importance = importance
        self.is_case_attribute = is_case_attribute
