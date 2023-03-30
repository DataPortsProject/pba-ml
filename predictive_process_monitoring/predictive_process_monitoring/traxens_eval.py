from predictive_process_monitoring.traxens_train import TraxensModelConf1


from predictive_process_monitoring.prbpm_models.evaluation import Evaluator

if __name__ == "__main__":
    Evaluator(TraxensModelConf1()).plot_cumulative_mcc(min_size=10, save=True)