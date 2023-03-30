import logging
from predictive_process_monitoring.prbpm_models.evaluation import Evaluator
from predictive_process_monitoring.prbpm_models.inference import InferenceModel
from dataset_definition.valenciaPort import ValenciaPort, VPModelConfiguration
import os
from os.path import join

class ValenciaPortModel(InferenceModel):
    def __init__(self):
        try:
            dataset = ValenciaPort.from_pickle(join('data', 'ValenciaPort', 'VP_clean.pkl'))
        except:
            dataset = ValenciaPort.from_csv(join('data', 'ValenciaPort', 'VP_clean.csv'))

        super(ValenciaPortModel, self).__init__(VPModelConfiguration(dataset, load_stored_model=True, numOfModels=10))




