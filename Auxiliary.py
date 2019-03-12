import numpy as np
import csv
import cPickle


class Med2Vec:
    def __init__(self):
        self.patient_events_input_file = ''
        self.med2vec_sequence_output_file = ''

    # transform the raw event data to the form med2vec requires
    def events2sequences(self, patient_events_input_file, med2vec_sequence_output_file):
        self.patient_events_input_file = patient_events_input_file
        self.med2vec_sequence_output_file = med2vec_sequence_output_file

        patient_events_input = open(self.patient_events_input_file)
        patient_events_csv_reader = csv.reader(patient_events_input)

        final_output_sequence = []
        current_time_window_sequence = []
        delimiter_list = [-1]
        last_subject_id = ''
        last_hadm_id = ''
        last_time_window_number = ''

        for line in patient_events_csv_reader:
            current_subject_id = line[0]
            current_hadm_id = line[1]
            current_time_window_number = line[2]
            current_event = int(line[3])

            if current_hadm_id == last_hadm_id:  # the same patient
                if current_time_window_number == last_time_window_number:  # the same time window
                    current_time_window_sequence.append(current_event)
                else:  # different time window
                    current_time_window_sequence = list(set(current_time_window_sequence))  # remove duplicates
                    if current_time_window_sequence:
                        final_output_sequence.append(list(current_time_window_sequence))
                    current_time_window_sequence = []
                    current_time_window_sequence.append(current_event)
            else:  # a different patient
                current_time_window_sequence = list(set(current_time_window_sequence))  # remove duplicates
                if current_time_window_sequence:
                    final_output_sequence.append(list(current_time_window_sequence))
                    final_output_sequence.append(delimiter_list)
                current_time_window_sequence = []
                current_time_window_sequence.append(current_event)

            last_subject_id = current_subject_id
            last_hadm_id = current_hadm_id
            last_time_window_number = current_time_window_number

        if current_time_window_sequence:
            current_time_window_sequence = list(set(current_time_window_sequence))  # remove duplicates
            final_output_sequence.append(list(current_time_window_sequence))

        patient_events_input.close()
        cPickle.dump(final_output_sequence, open(self.med2vec_sequence_output_file, 'wb'))

    # compute code representation based on med2vec model, and write the representations to file
    def compute_code_representation_to_file(self, model_file_path, code_rep_output_file_path, code_size=2868):
        model_parameter = np.load(model_file_path)
        W_emb = model_parameter['W_emb']
        W_output = model_parameter['W_output']
        W_hidden = model_parameter['W_hidden']
        b_emb = model_parameter['b_emb']
        b_output = model_parameter['b_output']
        b_hidden = model_parameter['b_hidden']
        code_rep = {}
        for code in range(0, code_size):
            x_t = np.zeros(code_size)
            x_t[code] = 1
            rep_vector = np.dot(x_t, W_emb) + b_emb
            rep_vector = np.maximum(rep_vector, 0)
            code_rep[code] = rep_vector
        cPickle.dump(code_rep, open(code_rep_output_file_path, 'wb'))




class DBNProcessing:
    def dbn_processing(self, input_file, output_file, data_type='chartevents'):
        data_file = open(input_file)
        data_csv_reader = csv.reader(data_file)
        banjo_output_writer = open(output_file, 'w')

        chartevents_variables = [646, 778, 780, 220045, 220050, 220051, 220210, 220739, 223762, 223835, 223900, 223901]
        labevents_variables = [50882, 50885, 50893, 50902, 50912, 50931, 50971, 50983, 51006, 51221, 51301]
        all_processing_variables = []  # variables will be processed

        if data_type == 'chartevents':
            normal_status = self.define_normal_status(chartevents_variables)
            all_processing_variables = chartevents_variables
        else:
            normal_status = self.define_normal_status(labevents_variables)
            all_processing_variables = labevents_variables

        joined_headers = ' '.join([str(variable) for variable in all_processing_variables])
        banjo_output_writer.write("%s\n" % joined_headers)
        previous_output_status = normal_status.copy()
        previous_patient = ''
        previous_window_number = -1
        current_output_status = {}
        for line in data_csv_reader:
            subject_id = line[0]
            hadm_id = line[1]
            itemid = int(line[2])
            current_window_number = int(line[3])
            current_window_status = line[4]
            current_patient = subject_id + ',' + hadm_id
            if current_patient == previous_patient:  # the same patient
                if current_window_number == previous_window_number:  # the same time window
                    current_output_status[itemid] = current_window_status
                else:  # a different time window
                    # impute missing variables with previous window's value
                    for variable in all_processing_variables:
                        if variable not in current_output_status.keys():
                            current_output_status[variable] = previous_output_status.get(variable)
                    # output the data in required format
                    joined_current_output_status = ' '.join(
                        [str(value) for value in [current_output_status.get(key) for key in all_processing_variables]])
                    banjo_output_writer.write("%s\n" % joined_current_output_status)

                    if current_window_number - previous_window_number != 1:
                        for i in range(0, current_window_number - previous_window_number - 1):
                            banjo_output_writer.write("%s\n" % joined_current_output_status)

                    previous_window_number = current_window_number
                    previous_output_status = current_output_status.copy()
                    current_output_status.clear()
                    current_output_status[itemid] = current_window_status

            else:  # a different patient
                # handle the last window of the previous patient
                if previous_patient != '':
                    for variable in all_processing_variables:
                        if variable not in current_output_status.keys():
                            current_output_status[variable] = previous_output_status.get(variable)
                    joined_current_output_status = ' '.join(
                        [str(value) for value in [current_output_status.get(key) for key in all_processing_variables]])
                    banjo_output_writer.write("%s\n" % joined_current_output_status)
                previous_output_status = normal_status.copy()
                current_output_status.clear()
                current_output_status[itemid] = current_window_status
                previous_patient = current_patient
                previous_window_number = current_window_number

        for variable in all_processing_variables:
            if variable not in current_output_status.keys():
                current_output_status[variable] = previous_output_status.get(variable)
        joined_current_output_status = ' '.join(
            [str(value) for value in [current_output_status.get(key) for key in all_processing_variables]])
        banjo_output_writer.write("%s\n" % joined_current_output_status)

        banjo_output_writer.close()

    def define_normal_status(self, variables):
        glasgow_coma_scale_normal = {220739: 4, 223900: 5, 223901: 6}
        normal_status = {}
        for single_variable in variables:
            normal_status[single_variable] = 2
        for single_variable in glasgow_coma_scale_normal.keys():
            if single_variable in variables:
                normal_status[single_variable] = glasgow_coma_scale_normal.get(single_variable)
        return normal_status
