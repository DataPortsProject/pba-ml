from .loreley import DatasetConfig
from .loreley import Loreley
from predictive_process_monitoring.VP.valencia_port_inference import ValenciaPortModel
import time
import numpy as np
import os

os.chdir(os.getcwd())

model = ValenciaPortModel()

dataset = model.get_dataset()

# As ActivityId is the feature that tells the event type and is the first feature in the vector
event_type_feature_index = 0
metadata = dataset.test_set[0].metadata

for entry in metadata.values():
    if entry['index'] == event_type_feature_index:
        entry['is_event_id'] = True
    else:
        entry['is_event_id'] = False

cutoff = 1
dataset_config = DatasetConfig(
    #should_one_hot_encode_event_types=True,
    #metadata=metadata,
    #cutoff=cutoff,
    #ga_number_of_generations=20,
    metadata=metadata,
    ga_number_of_generations=20,
    ga_target_size=100,
    mutation_chance=0.1,
    case_attribute_columns=False,
    initial_timestamp=False,
    transform_timestamps=False,
    use_packing=True
)
loreley = Loreley(model, dataset, dataset_config=dataset_config)


over_all_start = time.time()
current_same_predictions = 0


# define trace IDs you want to explain
traces_of_interest = [1, 2, 5]

for n, i in enumerate(traces_of_interest):

    trace = dataset.test_set[0].traces[i]

    trace_len = np.shape(trace)[0]
    t = trace.copy()
    loreley.string_decode_categorical_integers([t])
    model_prediction = model.predict_binary_class_for_traces([t[:(-1) * cutoff]])
    actual_outcome = dataset.test_set[1][i]

    print("Explaining [" + str(n + 1) + "/" + str(len(traces_of_interest)) + "]")
    print("The trace length is " + str(trace_len) + " with a prefix length of " + str(trace_len - cutoff))

    if model_prediction[0] == actual_outcome:
        print("The trace was predicted correctly by the bb model")
    else:
        print("The trace was not predicted correctly by the bb model")

    start_time = time.time()
    explanation, fidelity, same = loreley.get_explanation_for_trace(trace, cutoff=1)
    end_time = time.time()
    if same:
        current_same_predictions += 1
    explanation.print_explanation()
    print("The local model has a fidelity of " + str(fidelity))
    if same:
        print("The local model predicted the trace correctly")
    else:
        print("The local model did not predict the trace correctly")
    print("This puts the current hit score are at " + str(current_same_predictions / (n + 1)))
    print("Prediction took " + str(end_time - start_time) + " seconds.")

    vis_arr = np.zeros(shape=np.shape(explanation.explanation_trace))
    for imp in explanation.feature_importances:
        vis_arr[imp.event, imp.feature] = imp.importance

    lp = 'non-violation' if explanation.prediction == 0 else 'violation'


print("Process took " + str(end_time - over_all_start) + " seconds.")
