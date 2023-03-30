# Process Based Analytics

## Description

The Process-based Analytics component is part of the Advanced Big Data Analytics layer of the DataPorts platform. It analyses business processes by using both historic and real-time data available inside the DataPorts platform to provide its predictive results to cognitive applications, which inform the end-users about the predictions. 
It consists of the following three main components. By exploiting advanced data analytics techniques and machine learning, these components offer decision support for terminal and process operators, thereby facilitating proactive management of port processes:

- Ensemble Predictive Process Monitoring: This component uses ensembles of deep learning models (recurrent neural networks) to provide accurate predictions for each point during process execution, i.e., in a streaming fashion. 

- Prescriptive Process Monitoring: Building on process predictions, Online Reinforcement Learning allows automating the process on when to adapt a running process. We apply state-of-the-art Reinforcement Learning algorithms to the problem of identifying the signs of possible failure early and accurately. 

- Explainable Predictive Process Monitoring: This component aims at providing interpretations on why a certain prediction is made by a black-box predictive model, in particular by the deep learning models used in the first component above. To generate highly accurate predictions and at the same time facilitate interpretability for predictive process monitoring tasks, we leverage the concept of model induction from interpretable machine learning (ML) research. 

## Installation

Required is an installation of python 3. We recommend using the Anaconda tool to create a virtual environment. 

In your environment (system-wide or virtual) run "pip install -v -r requirements.txt" in this porject's directory. This installs all necessary requiremnts, which may take a long time, up to several hours. Once the installation process is done, run "pip install -v -e prescriptive_process_monitoring\gym-threshold-begone" to install said directory as a custom pip library. Adjust the directory separator in accordance to your operating system. 

## Running

To run the entire component, simply execute the advanced_analytics_component.py script. You can modify its behaviour by editing the c.json configuration file. 
Note that running with the default configuration may take forever, up to several dozen hours. You can modify the configuration to run faster, if you want to just test your installation. Set the "upto" parameter to 5, in order to set the ensemble size. Set "max_epochs" to 2, in order to limit the training of  the ensemble to 2 repeats of the training-data. Multiply "bagging_size", "validationdata_split" and "testdata_split" by 1/1000 to drastically reduce the amount of data trained with. 

## Results

Running the advanced_analytics_component.py script will generate predictions (single inductor predictions and ensemble predictions) in a new predictions folder, as well as prescriptions in a new prescriptions folder. 