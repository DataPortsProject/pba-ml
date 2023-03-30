from predictive_process_monitoring.prbpm_models.evaluation import Evaluator
import predictive_process_monitoring.VP.valencia_port_inference
from predictive_process_monitoring.VP.valencia_port_inference import ValenciaPortModel

def run_eval(dataset_name):
    if dataset_name == "traxens":
        print("running traxens_train...")
        import predictive_process_monitoring.traxens_eval
    elif dataset_name == "pro":
        print("running pro train...")
        import predictive_process_monitoring.pro_train
    elif dataset_name == "valencia_port":
        print("running valencia_port eval...")
        Evaluator(ValenciaPortModel()).plot_cumulative_mcc(min_size=3)