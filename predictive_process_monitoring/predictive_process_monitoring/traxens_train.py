from dataset_definition.traxens import TraxensDataset, TraxensModelConf1
from predictive_process_monitoring.prbpm_models.model import EnsembleModel
from os.path import join, dirname, realpath

dataset = TraxensDataset.from_csv(join('data', 'traxens', 'table_combined.csv'))
model = TraxensModelConf1(dataset, load_stored_model=False)
model.train_model()
