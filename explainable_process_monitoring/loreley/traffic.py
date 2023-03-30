from utils import get_number_of_different_values_for_feature
from loreley import DatasetConfig
from loreley import Loreley
from classification_models.traffic import TrafficModel
import time
import numpy as np

# from visualization.explain_table_loreley import ExplainTable

model = TrafficModel()

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

# explain_table = ExplainTable(TrafficModel(), ['non-violation', 'violation'])

traces_of_interest = [0, 1, 2, 3]

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

    # explain_table.save_to_file("plots/Traffic_Trace_" + str(i) + ".pdf",
    #                            i, trace_len - cutoff, vis_arr, local_prediction=float(explanation.prediction),
    #                            fidelity=fidelity, only_save_args=True)

print("Process took " + str(end_time - over_all_start) + " seconds.")
