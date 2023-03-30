import logging
import os
from abc import ABC
from os import listdir
from os.path import join, isfile, dirname
from typing import Generator
import re

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from predictive_process_monitoring.prbpm_models.dataset import Dataset, Batch, pre_zero_pad, pre_zero_pad_dataset, standardize_all_columns, one_hot_encode_all_columns, mask_nan_all_columns
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import sys

tensorflow.get_logger().setLevel(logging.ERROR)


class ClassificationModel(ABC):
    configuration_name = ''

    def __init__(self, dataset: Dataset, load_stored_model=False):
        self.dataset = dataset
        self.model = self._load_stored_model() if load_stored_model else None

    def is_keras_model(self):
        return True

    def generator(self, batch_size) -> Generator:
        dataset_generator = self._get_training_data().generator(batch_size=batch_size)
        while True:
            prepared_dataset = self.prepare_dataset(next(dataset_generator))
            yield np.array(prepared_dataset.dataset[0].traces).astype(float), \
                  np.array(prepared_dataset.dataset[1])
                

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return pre_zero_pad_dataset(dataset, extend_labels=False)

    def train_model(self, batch_size=128, steps_per_epoch=30, epochs=10000, callbacks=()):
        self.model = self._build_model()

        model_checkpoint = ModelCheckpoint(
            '%s/stored_models/model_%s_loss_{loss:.4f}_epoch_{epoch:02d}.h5'
            #% (str(sys.path[0]), self.configuration_name, str(i)),  # sys.path[0] is the current working directory
            % (dirname(sys.modules[self.__module__].__file__), self.configuration_name),
            save_best_only=True,
            monitor='loss'
        )

        self.model.fit(
            self.generator(batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=10_000,
            verbose=2,
            batch_size=batch_size,
            callbacks=[model_checkpoint, *callbacks],
        )

    def predict_probabilities(self, batch: Batch):
        assert batch.metadata == self.dataset.metadata, 'Metadata of batch have to match the dataset metadata.'
        # assert self.model is not None, 'No model currently stored. Run train_model() first.'

        prepared_batch = self._prepare_batch_columns(batch)
        return self._predict(prepared_batch)

    def _predict(self, prepared_batch: Batch) -> list[float]:
        batch_as_np_array = np.array(pre_zero_pad(prepared_batch).traces).astype('float32')
        raw_predictions = self.model.predict(batch_as_np_array)
        if np.ndim(raw_predictions) == 3:
            return np.array([prediction[-1][0] for prediction in raw_predictions])
        else:
            return np.ravel(raw_predictions)

    def _load_stored_model(self):
        path_to_models = join(dirname(sys.modules[self.__module__].__file__), 'stored_models')
        available_model_files = [f for f in listdir(path_to_models)
                                 if isfile(join(path_to_models, f)) and f.startswith(f'model_{self.configuration_name}_')]

        if len(available_model_files) == 0:
            return None

        best_model_file = sorted(available_model_files)[0]
        return load_model(join(path_to_models, best_model_file))

    @staticmethod
    def _build_model():
        raise NotImplementedError

    def _prepare_batch_columns(self, batch: Batch) -> Batch:
        if self.is_keras_model():
            batch = standardize_all_columns(batch)
            batch = one_hot_encode_all_columns(batch)
            batch = mask_nan_all_columns(batch)
        return batch

    def _get_training_data(self) -> Dataset:
        prepared_batch = self._prepare_batch_columns(self.dataset.training_set[0])
        prepared_dataset = Dataset(prepared_batch.traces, self.dataset.training_set[1],
                                   prepared_batch.metadata)
        return prepared_dataset

class EnsembleModel(ClassificationModel):
    def __init__(self, dataset: Dataset, load_stored_model=True, numOfModels=10):
        self.dataset = dataset
        self.numOfModels = numOfModels
        self.models = self._load_stored_models() if load_stored_model else [None for _ in range(0,numOfModels)]

    def is_keras_model(self):
        return True

    def generator(self, batch_size) -> Generator:
        dataset_generator = self._get_training_data().generator(batch_size=batch_size)
        while True:
            prepared_dataset = self.prepare_dataset(next(dataset_generator))
            yield np.array(prepared_dataset.dataset[0].traces).astype(float), \
                  np.array(prepared_dataset.dataset[1]).astype(float)

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return pre_zero_pad_dataset(dataset, extend_labels=False)

    def train_model(self, batch_size=128, steps_per_epoch=30, epochs=10000, callbacks=()):
        for i in range(0, self.numOfModels):
            currentModel = self._build_model()
            self.models[i] = currentModel

            model_checkpoint = ModelCheckpoint(
                '%s/stored_models/%s/ensemble_%s_epoch_{epoch:02d}.h5'
                % (str(sys.path[0]), self.configuration_name, str(i)), # sys.path[0] is the current working directory
                save_best_only=True,
                monitor='loss'
            )

            currentModel.fit(
                self.generator(batch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=2,
                batch_size=batch_size,
                callbacks=[model_checkpoint, *callbacks],
            )


    def predict_probabilities(self, batch: Batch):
        assert batch.metadata == self.dataset.metadata, 'Metadata of batch have to match the dataset metadata.'
        # assert self.model is not None, 'No model currently stored. Run train_model() first.'

        prepared_batch = self._prepare_batch_columns(batch)
        return self._predict(prepared_batch)

    def _predict(self, prepared_batch: Batch) -> list[float]:
        predictions = [self._predict_single(prepared_batch, model) for model in self.models]
        mean_predictions = np.array(predictions).mean(axis=0)
        return mean_predictions

    def _predict_single(self, prepared_batch: Batch, model) -> list[float]:
        batch_as_np_array = np.array(pre_zero_pad(prepared_batch).traces).astype('float32')
        raw_predictions = model.predict(batch_as_np_array)
        if np.ndim(raw_predictions) == 3:
            return np.array([prediction[-1][0] for prediction in raw_predictions])
        else:
            return np.ravel(raw_predictions)

    def _load_stored_models(self):
        path_to_models = join(os.path.dirname(sys.argv[0]), 'stored_models', self.configuration_name)
        available_model_files = [f for f in listdir(path_to_models)
                                 if isfile(join(path_to_models, f)) and f.startswith(f'ensemble_')]

        if len(available_model_files) == 0:
            raise Exception("No Model files available")

        epoch_matches = [re.search(r'epoch_(\d+)', file) for file in available_model_files]
        epochs = [e.group(1) for e in epoch_matches]
        epochs_sorted = sorted(epochs)
        last_epoch = epochs_sorted[-1]
        models = [load_model(join(path_to_models, "ensemble_" + str(i) + "_epoch_" + last_epoch + ".h5")) for i in range(self.numOfModels)]
        return models

    @staticmethod
    def _build_model(model_idx):
        raise NotImplementedError

    def _prepare_batch_columns(self, batch: Batch) -> Batch:
        if self.is_keras_model():
            batch = standardize_all_columns(batch)
            batch = one_hot_encode_all_columns(batch)
            batch = mask_nan_all_columns(batch)
        return batch

    def _get_training_data(self) -> Dataset:
        prepared_batch = self._prepare_batch_columns(self.dataset.training_set[0])
        prepared_dataset = Dataset(prepared_batch.traces, self.dataset.training_set[1],
                                   prepared_batch.metadata)
        return prepared_dataset