import csv
import pandas as pd
import numpy as np
import math
import time
import sys


def training_trans(txt):
    out = open("training.csv", 'w', newline='')
    csv_writer = csv.writer(out, dialect='excel')
    f = open(txt, "r")
    for line in f.readlines():
        line = line.replace(',', '\t')  # change every "," into " "
        list = line.split()  # translate string into list, write into csv
        csv_writer.writerow(list)
    return out


def testing_trans(txt):
    out = open("testing.csv", 'w', newline='')
    csv_writer = csv.writer(out, dialect='excel')
    f = open(txt, "r")
    for line in f.readlines():
        line = line.replace(',', '\t')  # change every "," into " "
        list = line.split()  # translate string into list, write into csv
        csv_writer.writerow(list)
    return out


def write_raw_index(file):  # add index at the top of csv
    filename = file
    with open(filename, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        text = 'Location' + ',' + 'MinTemp' + ',' + 'MaxTemp' + ',' + 'Rainfall' + ',' + 'Evaporation' + ',' + 'Sunshine' + ',' \
               + 'WindGustDir' + ',' + 'WindGustSpeed' + ',' +'WindDir9am' + ',' + 'WindDir3pm' + ',' + 'WindSpeed9am' + ',' \
               + 'WindSpeed3pm' + ',' + 'Humidity9am' + ',' + 'Humidity3pm' + ',' + 'Pressure9am' + ',' + 'Pressure3pm' + ',' \
               + 'Cloud9am' + ',' + 'Cloud3pm' + ',' + 'Temp9am' + ',' + 'Temp3pm' + ',' + 'RainToday' + ',' + 'RainTomorrow'
        f.write(text + '\n' + content)


def prior_prob(train_data):  # Calculate Prior probabilities
    tot_rows = float(train_data.shape[0])
    yes_count = train_data[train_data['RainTomorrow'] == 'Yes'].shape[0]
    no_count = train_data[train_data['RainTomorrow'] == 'No'].shape[0]
    prior_yes_prob = yes_count / tot_rows  # P(target = yes)
    prior_no_prob = no_count / tot_rows  # P(target = no)
    return prior_yes_prob, prior_no_prob, tot_rows


def calculate_prob(mean, std, x):  # Continuous data probability
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp


def calculate_discrete_variable_prob(attr, val, tot_rows):  # discrete data probability
    y_count = train_data[(train_data['RainTomorrow'] == 'Yes') & (train_data[attr] == val)].shape[0]
    n_count = train_data[(train_data['RainTomorrow'] == 'No') & (train_data[attr] == val)].shape[0]
    return y_count / tot_rows, n_count / tot_rows


def predict(test_data, prior_yes_prob, prior_no_prob):
    continuous_attr_yes_dict = {}  # Calculate mean and std for each attribute w.r.t class label
    continuous_attr_no_dict = {}
    # decide which kind of data is continuous data
    # by determining if the data type is object. If not, the data type is continuous.
    continuous_attributes = list(train_data.dtypes[train_data.dtypes != 'object'].index)
    for attribute in continuous_attributes:
        np_arr_yes = np.array(train_data[train_data['RainTomorrow'] == 'Yes'][attribute])
        np_arr_no = np.array(train_data[train_data['RainTomorrow'] == 'No'][attribute])
        continuous_attr_yes_dict[attribute] = (np.mean(np_arr_yes), np.std(np_arr_yes))
        continuous_attr_no_dict[attribute] = (np.mean(np_arr_no), np.std(np_arr_no))

    predict_file = open("predict.txt","w")
    predict_label = []
    correct_predictions = 0
    wrong_predictions = 0
    test_records = test_data.shape[0]

    for i in range(test_records):
        row = test_data.iloc[i]
        yes_prob = 1
        no_prob = 1
        for attribute, value in row.iteritems():
            if attribute in continuous_attr_yes_dict:
                yes_mean = continuous_attr_yes_dict.get(attribute)[0]
                yes_std = continuous_attr_yes_dict.get(attribute)[1]
                yes_prob *= calculate_prob(yes_mean, yes_std, value)
                no_mean = continuous_attr_no_dict.get(attribute)[0]
                no_std = continuous_attr_no_dict.get(attribute)[1]
                no_prob *= calculate_prob(no_mean, no_std, value)
            else:
                discrete_probabilities = calculate_discrete_variable_prob(attribute, value, tot_rows)
                yes_prob *= discrete_probabilities[0]
                no_prob *= discrete_probabilities[1]
        # Multiply by prior probability
        yes_prob *= prior_yes_prob
        no_prob *= prior_no_prob

        predicted_class = 1 if yes_prob > no_prob else 0
        predict_file.write(str(predicted_class)+"\n")
        print(predicted_class)

        predict_label.append(predicted_class)
        actual_class = class_labels.values[i]

        if (predicted_class == 0 and actual_class == 'No') or (predicted_class == 1 and actual_class == 'Yes'):
            correct_predictions += 1
        else:
            wrong_predictions += 1
    predict_file.close()
    return correct_predictions, wrong_predictions, predict_label




if __name__ == '__main__':
    start = time.time()
    #training_trans(sys.argv[2])
    #testing_trans(sys.argv[1])
    training_trans("training.txt")  # choose a training data
    testing_trans("testing.txt")  # choose a testing data. if use training data as test data, it should be "training.txt"
    write_raw_index("training.csv")
    write_raw_index("testing.csv")

    train_data = pd.read_csv("C:\\Users\\13703\\Documents\\learning\\exchange\\cs165a\\mp1\\training.csv")  # Load training data
    test_data = pd.read_csv("C:\\Users\\13703\\Documents\\learning\\exchange\\cs165a\\mp1\\testing.csv")  # Load testing data


    # preparing for the test data: delete the label
    # Take class labels separately and remove it from test data
    class_labels = test_data['RainTomorrow']
    del test_data['RainTomorrow']

    prior_yes_prob, prior_no_prob, tot_rows = prior_prob(train_data)
    correct_predictions, wrong_predictions, predict_label = predict(test_data, prior_yes_prob, prior_no_prob)
    predict(test_data, prior_yes_prob, prior_no_prob)


    accuracy = correct_predictions/ (wrong_predictions + correct_predictions)
    end = time.time()

    print('the running time is :{}'.format(end - start))


 #   print('the predict label is:')
  #  for j in range(len(predict_label)):
   #     print(predict_label[j], end=' ')
    #    print("")



    print("")
    print('the accuracy is:{}'.format(accuracy))










