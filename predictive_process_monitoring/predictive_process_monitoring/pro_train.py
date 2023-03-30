from dataset_definition.traxens import TraxensDataset
from predictive_process_monitoring.prbpm_models.model import EnsembleModel
from os.path import join, dirname, realpath

dataset = TraxensDataset.from_pickle(join('data', 'traxens', 'table_combined.pkl'))
model = EnsembleModel(dataset)
model.train_model()
