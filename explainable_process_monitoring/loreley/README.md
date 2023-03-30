# Loreley Implementation

This project is made and executed with Python 3.9.6

The Core implementation of Loreley can be found in [loreley.py](loreley.py).
It comes with some possibilites to configure.
The files [dummy.py](dummy.py), [cargo2000.py](cargo2000.py), [traffic.py](traffic.py), [bpic2012.py](bpic2012.py), and [bpic2017.py](bpic2017.py) show a sample usage of Loreley with a single data set.
Furthermore, the [config_quantification.py](config_quantification.py) shows a setup that was used for the quantitative evaluation of Loreley.
To execute any of these files the dependencies need to be installed. 
For that navigate to the loreley directory in the project and execute the following command:
````shell
pip install -r requirements.txt.txt
````
This installs the needed dependencies to the currently active python environment also needed by the black box model.

## General usage of Loreley API

For Loreley to work, a black box model that conforms with the interface defined in [inference.py](../classification_models/classification_models/inference.py) is required.
This needs to be instantiated first.
Since it is required for Loreley to be process aware, we also need to mark the event label index in the data set.
We do that in the provided metadata by the [interface](../classification_models/classification_models/inference.py) of the provided data set.
E.g. for BPIC 2017 the event id index is 2. 
Wherever the event index is not equal to 2, the field "is_event_id" needs to be set to False in the metadata, and True, where it is equal to 2.
A sample implementation can be seen here:
``` python
event_type_feature_index = 2
metadata = model.get_dataset().test_set[0].metadata

for entry in metadata.values():
    if entry['index'] == event_type_feature_index:
        entry['is_event_id'] = True
    else:
        entry['is_event_id'] = False
```
The Loreley script furthermore, needs a DatasetConfig defined in [loreley.py](loreley.py).
It takes the adjusted metadata, parameters for the genetic algorithm (ga_number_of_generations, ga_target_size, mutation_chance, crossover_chance), and parameters for the method improvements (use_packing, transform_timestamps, initial_timestamp, case_attribute_columns).
However, only the metadata is required and all other parameters have sensible default values.
Moreover, all improvement methods are turned on by default.

After Loreley is configured it can be set up and the get_explanation_for_trace method can be called for a trace in the form of traces provided by the test data set.
It returns three values. The explanation, the fiedlity, and if the local model produced the same result as the black box model.
While the fidelity and the same prediction values contain only the values, the explanation has a form defined in [loreley.py](loreley.py).
The explanation consists of the explanation trace, the local model prediction, a list of explanation terms, and a list of feature importances.
Each explanation term consists of meta data about the explanation term (e.g. categorical or numerical...) and each Feature importance consists of the event and feature id, the importance (which is the weighted gini impurity), and a boolean if it is a case attribute.
The method print_explanation prints the explanation to the console in a human readable form.
The method log_explanation takes a logger, which then prints and saves the explanation to a human readable form.
Both printing mechanisms print the explanation in a LaTeX friendly format, so that it can just be copied over into a LaTeX document.

## Quantitative Usage

The [config_quantification.py](config_quantification.py) is mainly used to run multiple configurations at once.
It is tailored towards being configured from the command line.

It takes the dataset that should be run as a first argument ('cargo2000', 'c2kstripped', 'traffic', 'bpic2012', 'bpic2017', 'dummy'), as a second argument it takes 'short' or 'long' to choose the type of traces for a run, and lastly it takes a run_number to prefix files, so they do not get overridden when performing multiple runs.
Furthermore, to save the results, the folders results and top_features_results need to be created in the directory the script is executed in.
A sample shell script executing 4 different configurations for traffic data set can be seen here:
```shell
python config_quantification.py traffic short 0; \
python config_quantification.py traffic short 1; \
python config_quantification.py traffic long 0; \
python config_quantification.py traffic long 1;
```
Furthermore, a target_prefix_length is determined for each data set configuration (either long or short).
Each entered traces gets then cutoff to get to the target length.
In the case of the cargo2000 model the traces get cut to a prefix length of 6.

The created csvs in the results folder contain the following columns: ['trace_id', 'actual_outcome', 'black_box_correct', 'local_align_with_bb', 'fidelity', 'exp_feature_1',
                  'exp_feature_1_value', 'exp_feature_1_threshold', 'exp_feature_2', 'exp_feature_2_value',
                  'exp_feature_2_threshold', 'exp_feature_3', 'exp_feature_3_value', 'exp_feature_3_threshold',
                  'exp_feature_4', 'exp_feature_4_value', 'exp_feature_4_threshold', 'exp_feature_5',
                  'exp_feature_5_value', 'exp_feature_5_threshold'].
The created csvs in the top_features_results folder contain the following columns: top_5_features_cols = ['trace_id', 'f1_attr_idx', 'f1_event_idx', 'f2_attr_idx', 'f2_event_idx', 'f3_attr_idx',
                       'f3_event_idx', 'f4_attr_idx', 'f4_event_idx', 'f5_attr_idx', 'f5_event_idx']
