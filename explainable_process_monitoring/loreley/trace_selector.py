from classification_models.bpic_2012 import BPIC2012Model
from classification_models.bpic_2017 import BPIC2017Model
from classification_models.traffic import TrafficModel
from classification_models.cargo_2000 import Cargo2000Model
import random

dataset_bpic2012 = BPIC2012Model().get_dataset()
test_set_traces = dataset_bpic2012.test_set[0].traces
bpic2012_trace_length_3 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 3], 100)
print(bpic2012_trace_length_3)
print('-------------------------')
bpic2012_trace_length_10 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 10], 100)
print(bpic2012_trace_length_10)
print('-------------------------')

dataset_bpic2017 = BPIC2017Model().get_dataset()
test_set_traces = dataset_bpic2017.test_set[0].traces
dataset_bpic2017_trace_length_18 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 18], 100)
print(dataset_bpic2017_trace_length_18)
print('-------------------------')
dataset_bpic2017_trace_length_56 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 56], 100)
print(dataset_bpic2017_trace_length_56)
print('-------------------------')

dataset_traffic = TrafficModel().get_dataset()
test_set_traces = dataset_traffic.test_set[0].traces
traffic_trace_length_2 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 2], 100)
print(traffic_trace_length_2)
print('-------------------------')
traffic_trace_length_9 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 9], 100)
print(traffic_trace_length_9)
print('-------------------------')

dataset_c2k = Cargo2000Model().get_dataset()
test_set_traces = dataset_c2k.test_set[0].traces
c2k_trace_length_8 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 8], 100)
print(c2k_trace_length_8)
print('-------------------------')
c2k_trace_length_20 = random.sample([i for i in range(len(test_set_traces)) if test_set_traces[i].shape[0] == 20], 100)
print(c2k_trace_length_20)
print('-------------------------')
