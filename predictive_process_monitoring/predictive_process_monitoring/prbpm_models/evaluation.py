import math
from typing import Literal, List

import matplotlib.pyplot as plt
import tikzplotlib
from predictive_process_monitoring.prbpm_models.dataset import Batch
from predictive_process_monitoring.prbpm_models.inference import InferenceModel


class Evaluator:

    def __init__(self, model: InferenceModel):
        self.model = model

    @staticmethod
    def _calculate_mcc(predictions: List[Literal[0, 1]], labels: List[Literal[0, 1]]) -> float:  # range [-1, 1]
        assert len(predictions) == len(labels), 'Prediction list and label list have to be of the same length.'
        tp = sum([predictions[i] == labels[i] == 1 for i in range(len(predictions))])  # true positives
        tn = sum([predictions[i] == labels[i] == 0 for i in range(len(predictions))])  # true negatives
        fp = sum([predictions[i] == 1 and labels[i] == 0 for i in range(len(predictions))])  # false positives
        fn = sum([predictions[i] == 0 and labels[i] == 1 for i in range(len(predictions))])  # false negatives

        def color(assertion):
            return f'\33[{(91 if assertion else 92)}m'

        print(f'\33[39mTP: {100 * tp / sum((tp, tn, fp, fn)): >6.2f}%% |',
              f'TN: {100 * tn / sum((tp, tn, fp, fn)): >6.2f}% |',
              f'FP: {100 * fp / sum((tp, tn, fp, fn)): >6.2f}% |',
              f'FN: {100 * fn / sum((tp, tn, fp, fn)): >6.2f}% |',
              color(tp + tn < fp + fn), f'Accuracy: {100 * sum((tp, tn)) / sum((tp, tn, fp, fn)): >6.2f}% |',
              color(tp < fn), f'Sensitivity: {100 * (tp / sum((tp, fn)) if sum((tp, fn)) > 0 else -0.01): >6.2f}% |',
              color(tn < fp), f'Specificity: {100 * (tn / sum((tn, fp)) if sum((tn, fp)) > 0 else -0.01): >6.2f}% |',
              color(tp < fp), f'Precision: {100 * (tp / sum((tp, fp)) if sum((tp, fp)) > 0 else -0.01): >6.2f}% |',
              f'\33[39mPredicted Violations: {100 * sum((tp, fp)) /sum((tp, tn, fp, fn)): >6.2f}% |',
              f'\33[39mActual Violations: {100 * sum((tp, fn)) /sum((tp, tn, fp, fn)): >6.2f}% |',
              f'\33[39mSample size: {len(predictions)} | Violation Size: {tp + fn}')

        denominator = math.sqrt(tp + fp) * math.sqrt(tp + fn) * math.sqrt(tn + fp) * math.sqrt(tn + fn)
        return (tp * tn - fp * fn) / denominator if denominator != 0.0 else math.nan

    @staticmethod
    def _get_labeled_prefix_datasets(batch: Batch, labels: list, keep_full=True):
        traces = batch.traces
        max_trace_length = max([len(trace) for trace in traces])
        prefix_datasets = [(Batch([], batch.metadata), []) for l in range(max_trace_length)]

        for l in range(max_trace_length):
            l_batch = Batch([trace[:l + 1] for trace in traces if (keep_full and len(trace) > l) or len(trace) > l+1], batch.metadata)
            l_labels = [labels[i] for i, trace in enumerate(traces) if (keep_full and len(trace) > l) or len(trace) > l+1]
            prefix_datasets[l] = (l_batch, l_labels)

        return prefix_datasets

    @staticmethod
    def _get_sorted_labeled_prefix_datasets(batch: Batch, labels: list):
        traces = batch.traces
        max_trace_length = max([len(trace) for trace in traces])
        prefix_datasets = [[(Batch([], batch.metadata), []) for i in range( l +1)] for l in range(max_trace_length)]

        for l in range(max_trace_length):
            l_batch = [trace for trace in traces if len(trace) == l+ 1]
            if len(l_batch) == 0:
                prefix_datasets[l] = []
            else:
                l_labels = [labels[i] for i, trace in enumerate(traces) if len(trace) == l + 1]
                for i in range(l + 1):
                    i_batch = Batch([trace[:i + 1] for trace in l_batch], batch.metadata)
                    prefix_datasets[l][i] = (i_batch, l_labels)

        return prefix_datasets

    def calculate_mcc_for_dataset(self, batch: Batch, labels: list):
        if len(labels) == 0:
            return 0
        predictions = self.model.predict_binary_class_for_traces(batch.traces)
        return self._calculate_mcc(predictions, labels)

    def calculate_mcc_for_datasets(self, datasets):
        return [self.calculate_mcc_for_dataset(*dataset) for dataset in datasets]

    def plot_mccs(self, max_length=-1, min_size=0, save=False):
        training_prefixes = self._get_sorted_labeled_prefix_datasets(*self.model.get_dataset().training_set)
        test_prefixes = self._get_sorted_labeled_prefix_datasets(*self.model.get_dataset().test_set)

        training_mccs = [None for _ in training_prefixes]
        test_mccs = [None for _ in test_prefixes]

        print('-'*230)
        print('Evaluation of the Model', self.model.model.configuration_name)
        print('-'*230)
        print('Evaluation on Training Set')
        print('-'*230)
        for i in range(len(training_prefixes)):
            if len(training_prefixes[i]) > 0 and len(training_prefixes[i][0][1]) > min_size:
                print('Evaluation on Training Samples of Length', i+1)
                training_mccs[i] = self.calculate_mcc_for_datasets(training_prefixes[i])
                print('-'*230)
        print('Evaluation on Test Set')
        print('-'*230)
        for i in range(len(test_prefixes)):
            if len(test_prefixes[i]) > 0 and len(test_prefixes[i][0][1]) > min_size:
                print('Evaluation on Test Samples of Length', i+1)
                test_mccs[i] = self.calculate_mcc_for_datasets(test_prefixes[i])
                print('-'*230)

        if max_length < 0:
            max_length = min((len(training_mccs), len(test_mccs)))
        else:
            max_length = min((max_length, len(training_mccs), len(test_mccs)))

        for i in range(max_length):
            contains_training = training_mccs[i] is not None
            contains_test = test_mccs[i] is not None
            if contains_test and contains_training:
                plt.figure()
                if contains_training:
                    plt.plot([j+1 for j in range(i+1)], training_mccs[i], label="Training Set MCC")
                if contains_test:
                    plt.plot([j+1 for j in range(i+1)], test_mccs[i], label="Test Set MCC")
                plt.xlabel("Earliness")
                plt.ylabel("MCC")
                plt.legend(loc="upper left")
                plt.title(self.model.model.configuration_name + " with traces of length " + str(i + 1) +
                          " and sample sizes (" +
                          (str(training_mccs[i][1]) if len(training_mccs) > i else str(0)) + "," +
                          (str(test_mccs[i][1]) if len(test_mccs) > i else str(0)) + ")")
                if save:
                    tikzplotlib.save("figures/" + self.model.model.configuration_name + " - " + str(i + 1) + ".tex")
                    plt.savefig("figures/" + self.model.model.configuration_name + " - " + str(i + 1) + ".pdf")
                    plt.show()
                else:
                    plt.title(self.model.model.configuration_name)
                    plt.show()

    def plot_cumulative_mcc(self, max_length=-1, min_size=0, save=False, keep_full=False):
        training_prefixes = self._get_labeled_prefix_datasets(*self.model.get_dataset().training_set, keep_full=keep_full)
        training_prefixes = [prefix for prefix in training_prefixes if len(prefix[1]) > min_size]
        test_prefixes = self._get_labeled_prefix_datasets(*self.model.get_dataset().test_set, keep_full=keep_full)
        test_prefixes = [prefix for prefix in test_prefixes if len(prefix[1]) > min_size]

        print('-'*230)
        print('Evaluation of the Model', self.model.model.configuration_name)
        print('-'*230)
        print('Evaluation on Training Set')
        training_mcc = self.calculate_mcc_for_datasets(training_prefixes)
        print('-'*230)
        print('Evaluation on Test Set')
        test_mcc = self.calculate_mcc_for_datasets(test_prefixes)
        print('-'*230)

        if max_length < 0:
            max_length = min((len(training_mcc), len(test_mcc)))
        else:
            max_length = min((max_length, len(training_mcc), len(test_mcc)))

        plt.plot([i+1 for i in range(max_length)], training_mcc[:max_length], label="Training Set MCC")
        plt.plot([i+1 for i in range(max_length)], test_mcc[:max_length], label="Test Set MCC")
        plt.xlabel("Prefix Length")
        plt.ylabel("MCC")
        plt.legend(loc="upper left")
        if save:
            tikzplotlib.save("figures/" + self.model.model.configuration_name + ".tex")
            plt.savefig("figures/" + self.model.model.configuration_name + ".pdf")
            plt.show()
        else:
            plt.title(self.model.model.configuration_name)
            plt.show()
