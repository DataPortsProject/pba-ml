import argparse
from predictive_process_monitoring import train_runner, evaluate_runner
from explainable_process_monitoring.loreley import explainable_runner

TRAIN_MODEL = "train_model"
EVAL_MODEL = "eval_model"
EXPLAIN_PREDICTION = "explain_prediction"

PRO = "pro"
TRAXENS = "traxens"
VALENCIAPORT = "valencia_port"
#not implemented for datasets below
C2K = "c2k"
BPIC2012 = "bpic2012"
BPIC2017 = "bpic2017"
TRAFFIC = "traffic"


parser = argparse.ArgumentParser(description="Process some integers")
parser.add_argument('-m', "--mode", type=str, choices=[TRAIN_MODEL, EVAL_MODEL, EXPLAIN_PREDICTION], required=True)
parser.add_argument("-d", "--dataset", type=str, choices=[PRO, VALENCIAPORT, C2K, BPIC2012, BPIC2017, TRAFFIC, TRAXENS])

args = parser.parse_args()

if args.mode == TRAIN_MODEL:
    train_runner.train_dataset(args.dataset)
if args.mode == EXPLAIN_PREDICTION:
    explainable_runner.run_loreley(args.dataset)
if args.mode == EVAL_MODEL:
    evaluate_runner.run_eval(args.dataset)

#if __name__ == '__main__':
#    explainable_runner.run_loreley(VALENCIAPORT)


