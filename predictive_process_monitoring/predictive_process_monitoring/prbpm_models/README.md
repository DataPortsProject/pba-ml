# LSTM Prediction Blackbox
This package contains a black box model that was used in the XAI4BPM Poject at the University of Duisburg-Essen in the Summer Term 2021 to compare explainability techniques. Since *Predicitive Business Process Monitoring (PrBPM)* was chosen as the application domain, and thereby predictions of sequential processes are made, the black box is a recurrent neural network, namely a *Long Short-Term Memory (LSTM)*. This document describes the structure and architecture of the overall system and a tutorial on how arbitrary PrBPM datasets can be used for training the black box.

## Architecture
![Class Diagram](https://i.imgur.com/Ym2CCj0.png)

In the architecture, inference, i.e., generating predictions on arbitrary inputs, and training are separated. The reason for this is that training should not be performed every time the model is instantiated. Instead, the inference model loads the weights of the best-trained model when instantiated. The actual training takes place separately in the `ClassificationModel` class and stores the best weights as a file. To avoid overfitting, a separate validation loss is used to select the best model instead of the training loss. Since many different hyperparameters need to be considered, there are several `Configurations` for each data set. These represent promising settings and include, besides the actual hyperparameters, different types of pre- or post-processing and different LSTM architectures for more information).

The `ClassificationModel` is trained on a `Dataset`. This dataset contains all available labeled samples, divided into a training set and a test set. In addition, the class contains metadata of the individual columns of the dataset. In particular, it is specified whether a column contains categorical or numerical data and whether a column has already been preprocessed (e.g., by *standardization* or *one-hot encoding*).

### Dataset
The dataset component provides a convenient base class for using the datasets. An instance of this class is also returned from the base interface by the method `get_dataset()`. To use this class, a subclass must be created for each dataset to account for dataset-specific properties. In such a subclass, it must first be specified how the dataset is broken down into individual traces (usually using a case ID). In addition, a labeling function must be created in order to specify whether a given trace is a violation or not. After successful labeling, the attributes that are no longer necessary must be removed to prevent information from being present during training that would not yet be present at runtime (so-called *information leakage*). Internally, the traces are split into a training and a test set in an adjustable two-thirds to one-third ratio. Another feature of the base class is to return the traces of the training set not all at once but as a generator that returns them batch-wise. This is useful when preprocessing significantly increases the memory footprint of the traces, and thus not all of the preprocessed training set would fit in memory. With the help of the generator, the batches can be preprocessed bit by bit, and the working memory can be released after training on a batch has been completed. 

In addition to traces and labels, an instance of the `Dataset` class also contains a dictionary with metadata containing information about its columns. Thereby, a distinction is made between numeric and categorical columns. For columns with numeric data, the metadata includes column names and column IDs as well as the mean and standard deviation of all values in the dataset. For columns with categorical data, the metadata contains, besides name and ID, a list of all unique values in the dataset. A special case of categorical columns are columns that have been one-hot encoded. In this case, a column is split into several sub-columns. Therefore, the metadata additionally contains a mapping from each categorical value to the column index representing this value.

To make the general handling of traces easier, a wrapper class for a collection of traces has been created. This class is called `Batch` and contains the metadata of its traces, similar to the `Dataset` class. For instances of this class, many useful helper functions were defined, which consistently adjust the metadata. Among others, these helper functions can perform a one-hot encoding of one or more columns and convert columns that are one-hot encoded back to their original representation. In addition, numeric columns can be normalized and standardized, and if there are traces of different lengths in a `Batch`, zero-padding (pre or post) can be used to bring them to the same length.

Another advantage of using the `Dataset` class and the `Batch` class is that, despite the absence of a static type system (as Python is used), runtime checks ensure that all traces are always in the correct format and dimensionality. The accompanying performance degradation is acceptable since only a prototypical implementation is provided.

### ClassificationModel
The `ClassificationModel` class contains the training logic and the trained weights and uses an instance of the previously described `Dataset` class. Analogous to the previously described class, the `ClassificationModel` is also a base class, which must be subclassed for each dataset and adapted to its requirements. Thereby, the subclasses must implement the method `build_model()`, in which the LSTM architecture is defined. Additionally, methods for preprocessing and post-processing of the samples (for training and inference) have to be implemented. Since all hyper-parameters are set in each subclass, different configurations can be tested by creating multiple subclasses for a single dataset.

During training, the best models, i.e., those with the lowest validation loss, are regularly stored. When the class is instantiated, the best model is then selected from these stored models to enable inference without retraining the model each time. 

For training and inference, the deep learning library *Tensorflow* and its high-level wrapper *Keras* are used. The training can be done either with the help of a complete dataset at once or successively using generators.

Since the `ClassificationModel` class contains the actual model and its associated weights, it also contains inference logic used by the next class described, the `InferenceModel`.

### InferenceModel
The `InferenceModel` class wraps the `ClassificationModel` to implement the two previously specified interfaces. Thus, the only additional logic that the `InferenceModel` contains is to convert the probability outputs generated by the `ClassificationModel` into binary values (necessary for the `BinaryClassificationModelInterface`).

### Evaluator
The `Evaluator` class is used to evaluate the model, and to do so, it accesses the `InferenceModel`. In order to evaluate Earliness, all traces are first converted into all possible prefixes. This means that a trace with length ten is converted into nine different prefixes, where each of these prefixes contains one more event than the previous one. The resulting prefixes are fed into the model, and from the resulting predictions, accuracy, sensitivity, specificity, and precision are calculated. Based on these metrics, the *MCC* can then be calculated, which is the primary measure of prediction accuracy in this project. The MCC is calculated once for the training set and once for the test set and then visualized in a diagram. Either the MCC can be displayed cumulatively or separately for each prefix length. Since the `Evaluator` was developed only with the provided interface in mind, it is agnostic and thus does not need to be adapted to each dataset.

## Tutorial
This tutorial provides the necessary steps to train the black box LSTM model on a new dataset from the domain of PrBPM. First, to group the files to  be created together, it is useful to create a new folder. Then, inside this folder, create the following four files: `dataset.py`, `model.py`, `inference.py`, and `__init__.py`. The content of each of these files has to follow a precise schema which is presented in the next four sections.

### `dataset.py`

Datasets are used and loaded by subclasses of the abstract *Dataset* class contained in the module *classificiation_models.dataset*. To use your own dataset, you, therefore, need to implement a class inheriting from the abstract *Dataset* class. This class needs to implement the following **static** methods:

* `delimiter()`: Returns the delimiter of the csv-file (generally, ',' or ';')
* `trace_column_name()`: Returns the name of the column that defines which rows belong to the same trace (trace id/case id)
* `_get_trace_label(trace: pandas.Dataframe)`: Returns the label of one trace given the trace itself, either 1, 0, or `None`. If `None` is returned, the trace is deleted as it is assumed to not have finished
* `categorical_columns()`: Returns a list containing the names of all categorical columns in the dataset
* `numerical_columns()`: Returns a list containing the names of all numerical columns in the dataset
* `case_attribute_columns()`: Returns a list containing the names of all case attribute columns in the dataset, i.e., attributes that cannot change over the course of one trace
* `timestamp_columns()`: Returns a list containing the names of all timestamp columns in the dataset or rather columns that must be monotonically increasing over the course of one trace
* `delete_columns()`: Returns a list containing the names of all columns that need to be deleted prior to the training and/or explaining. Usually the trace column and columns that are necessary for the trace labels

Note that the union of `numerical_columns()`, `categorical_columns()`, and `delete_columns()` should be a disjoint union containing all columns of the dataset.

### `model.py`

To implement (classification) models in our framework, a subclass of the abstract `ClassificationModel` class must be implemented. If the model uses the Keras Machine Learning Model library, the following two methods need to be implemented in the subclass:

* `_build_model()`: Static method that returns the Keras model
* `prepare_dataset(self, dataset: Dataset)`: Preparing the dataset for the training, for example, using the functions `ngram_dataset()` or `pre_zero_pad_dataset()` from the `prbpm_models.dataset` module (optional).

Alternatively, if the model should predict the outcome based on specific rules rather than a Keras model, one can also use override the following two functions:

* `is_keras_model(self)`: In this case, must return False
* `_predict(self, batch: Batch)`: Returns the predictions (as a `list[float]`) for this batch

Additionally, in both cases, the attribute `configuration_name` should be set as it is used for both save/load paths and as a title for the evaluation.

### `inference.py`
The inference model needs to load the best existent weights for the dataset and afterwards provide the ability to predict prior unseen traces using these weights. As the logic is already encapsulated in the base class, you only have to provide the constructor of the subclass as follows:
```python=
class DatasetNameModel(InferenceModel):
    def __init__(self):
        try:
            dataset = DatasetName.from_pickle(join(dirname(realpath(__file__)), 'datasets', 'DatasetName.pkl')) 
        except:
            dataset = DatasetName.from_csv(join(dirname(realpath(__file__)), 'datasets', 'DatasetName.csv'))
        super(DatasetNameModel, self).__init__(ModelConfigurationN(dataset))
```
Replace `DatasetName` with the name of the new dataset added and specify the ModelConfiguration you want to use by replacing `ModelConfigurationN` (e.g. by `ModelConfiguration1`).

### `__init__.py`
Finally stitch everything together and expose it to the outer world by modifying the `__init__.py` file as follows:
```python=
from .dataset import DatasetName
from . import model
from .inference import DatasetNameModel
```
Replace `DatasetName` with the name of the new dataset added.