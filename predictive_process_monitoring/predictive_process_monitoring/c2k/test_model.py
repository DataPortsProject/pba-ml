import logging
from os.path import join, dirname, realpath
from test_dataset import Cargo2000
from test_config import ModelConfiguration1
from predictive_process_monitoring.prbpm_models.evaluation import Evaluator
from predictive_process_monitoring.prbpm_models.inference import InferenceModel


class Cargo2000Model(InferenceModel):
    def __init__(self):
        dataset = Cargo2000.from_csv(join(dirname(realpath(__file__)), 'data', 'c2k.csv'))
        super(Cargo2000Model, self).__init__(ModelConfiguration1(dataset))


if __name__ == '__main__':
    Evaluator(Cargo2000Model()).plot_cumulative_mcc(min_size=10)