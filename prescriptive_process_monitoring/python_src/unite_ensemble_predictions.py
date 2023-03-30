from os import listdir
import unicodecsv
import pandas as pd


def unite(predictive_output_directory):
    childnames = listdir(predictive_output_directory)
    csv_paths = []
    for name in childnames:
        if name.endswith("results.csv"):
            csv_paths.append(predictive_output_directory + "/" + name)
    print("Found " + str(csv_paths.__len__()) + " single inductor prediction files!")
    csv_files = []
    for csv in csv_paths:
        csv_files.append(open(csv, "rb"))
    csv_reader = []
    for csv in csv_files:
        reader = unicodecsv.reader(csv, delimiter=',')
        csv_reader.append(reader)
        next(reader)  # skip header
    print("Opened all files...")
    line = ''
    case_ids = []
    process_lengths = []
    positions = []
    actual_durations = []
    planned_durations = []
    ensemble_predictions = []
    ensemble_deviations = []
    reliabilities = []
    count = 0
    while line is not None:
        if len(csv_reader) <= 0:
            raise Exception(
                "There should never be no readers available! Are there result-csv files in the " +
                "configurations prediction output directory?")
        predicted_durations = []
        prediction_deviations = []
        first_reader = True
        for reader in csv_reader:
            line = next(reader, None)
            if line is not None:
                if first_reader:
                    try:
                        case_id = int(line[0])
                        process_length = int(line[1])
                        position = int(line[2])
                        planned_duration = float(line[6])
                    except ValueError as e:
                        print("Parsing encountered an error: {}".format(e))
                        print("Stopping the read...")
                        line = None
                        break
                    actual_duration = line[5]
                    case_ids.append(case_id)
                    process_lengths.append(process_length)
                    positions.append(position)
                    if actual_duration.lower() == "true":
                        actual_duration = 1
                    elif actual_duration.lower() == "false":
                        actual_duration = 0
                    actual_duration = float(actual_duration)
                    actual_durations.append(actual_duration)
                    planned_durations.append(planned_duration)
                    first_reader = False
                predicted_duration = float(line[4])
                predicted_durations.append(predicted_duration)
                prediction_deviation = (predicted_duration - planned_duration) / planned_duration
                prediction_deviations.append(prediction_deviation)
        if line is not None:
            ensemble_prediction = 1. / len(predicted_durations) * sum(predicted_durations)
            ensemble_predictions.append(ensemble_prediction)
            ensemble_deviation = (ensemble_prediction - planned_duration) / planned_duration
            ensemble_deviations.append(ensemble_deviation)
            positive_prediction_deviations = [deviation for deviation in prediction_deviations if deviation > 0]
            majority_count = max(len(positive_prediction_deviations),
                                 len(prediction_deviations) - len(positive_prediction_deviations))
            reliability = majority_count / len(prediction_deviations)
            reliabilities.append(reliability)

        count = count + 1
        if count % 10000 == 0:
            print("Processed " + str(count) + " checkpoints...")
    print("Writing ensemble predictions...")
    columns = ['case_id',
               'process_length',
               'position',
               'actual_duration',
               'planned_duration',
               'ensemble_prediction',
               'ensemble_deviation',
               'reliability']
    data_zip = list(zip(
        *[case_ids, process_lengths, positions, actual_durations, planned_durations, ensemble_predictions,
          ensemble_deviations, reliabilities]))
    dataframe = pd.DataFrame(data_zip, columns=columns)
    dataframe.to_csv(predictive_output_directory + "/ensemble.csv", header=True, index=False)
    print("Closing files...")
    for csv in csv_files:
        csv.close()
    print("Ensemble predictions successfully united!")


if __name__ == '__main__':
    unite("../models-traffic-extracted")
