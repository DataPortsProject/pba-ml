import os
import numpy as np
from predictive_process_monitoring.prbpm_models.dataset import Dataset, Batch, pre_zero_pad, ngram_dataset, pre_zero_pad_dataset
from predictive_process_monitoring.prbpm_models.model import EnsembleModel
from test_dataset import Cargo2000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, BatchNormalization, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class ModelConfiguration1(EnsembleModel):
    configuration_name = 'experiment_1'

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return pre_zero_pad_dataset(dataset, extend_labels=True)

    @staticmethod
    def _build_model():
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(None, 260)))
        model.add(LSTM(300, return_sequences=True))
        model.add(LSTM(200, return_sequences=True))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam()
        loss = BinaryCrossentropy(from_logits=False)
        model.compile(loss=loss, optimizer=optimizer)

        return model



if __name__ == '__main__':
    from os.path import join, dirname, realpath
    c2k = Cargo2000.from_csv(join(dirname(realpath(__file__)), 'data', 'c2k.csv'))

    # model = ModelConfiguration1(c2k, load_stored_model=False)
    # model.train_model(epochs=2)

    model = ModelConfiguration1(c2k, load_stored_model=True)
    model.train_model(epochs=2)
