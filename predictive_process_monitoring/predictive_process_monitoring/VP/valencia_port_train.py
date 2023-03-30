from dataset_definition.valenciaPort import ValenciaPort, VPModelConfiguration
from os.path import join

dataset = ValenciaPort.from_csv(join('data', 'ValenciaPort', 'VP_clean.csv'))
model = VPModelConfiguration(dataset, load_stored_model=False, numOfModels=10)
model.train_model()

