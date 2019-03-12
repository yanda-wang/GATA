import csv
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random


class Triplet(nn.Module):

    def __init__(self, input_size=200, hidden_state_size=200, final_output_size=512,
                 dbn_matrix_file='data/dbn/dbn_matrix'):
        super(Triplet, self).__init__()

        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.final_output_size = final_output_size

        self.chartevents_itemids = [646, 778, 780, 220045, 220050, 220051, 220210, 220739, 223762, 223835, 223900,
                                    223901]
        self.labevents_itemids = [50882, 50885, 50893, 50902, 50912, 50931, 50971, 50983, 51006, 51221, 51301]

        reload_dbn_matrix = np.load(dbn_matrix_file)
        self.dbn_matrix_chartevents = torch.from_numpy(reload_dbn_matrix['dbn_matrix_chartevents']).float()
        self.dbn_matrix_labevents = torch.from_numpy(reload_dbn_matrix['dbn_matrix_labevents']).float()

        # define GRU cell for each variable
        self.GRU_Cell_646_SpO2 = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_778_PaCO2 = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_780_Arterial_PH = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_220045_Heart_Rate = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_220050_ABP_mean = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_220051_ABP_systolic = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_220210_Respiratory_Rate = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_220739_coma_scale_eye = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_223762_temperature = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_223835_fractional_inspired_oxygen = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_223900_coma_scale_verbal = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_223901_coma_scale_motor = nn.GRUCell(self.input_size, self.hidden_state_size)

        self.GRU_Cell_50882_Bicarbonate = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_50885_Bilirubin = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_50893_Calcium = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_50902_Chloride = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_50912_Creatinine = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_50931_Glucose = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_50971_Potassium = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_50983_Sodium = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_51006_BUN = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_51221_Hematocrit = nn.GRUCell(self.input_size, self.hidden_state_size)
        self.GRU_Cell_51301_white_blodd_cell = nn.GRUCell(self.input_size, self.hidden_state_size)

        self.fc_input_size = self.hidden_state_size * (len(self.chartevents_itemids) + len(self.labevents_itemids))
        self.fc = nn.Linear(self.fc_input_size, self.final_output_size)

    def forward_once(self, input_data_chartevents, input_data_labevents):
        hidden_state_chartevents = torch.zeros(len(self.chartevents_itemids), self.hidden_state_size,
                                               dtype=torch.float)
        for i, input_data_single_window in enumerate(
                input_data_chartevents.chunk(input_data_chartevents.size(0), dim=0)):
            split_single_window_input_data = torch.split(input_data_single_window, self.input_size, 1)

            cpt_medication_representation_vectors = split_single_window_input_data[12] + split_single_window_input_data[
                13]

            clone_hidden_state_chartevents = hidden_state_chartevents.clone()

            dbn_transfered_hidden_state_chartevents = torch.matmul(self.dbn_matrix_chartevents,
                                                                   clone_hidden_state_chartevents)

            hidden_state_chartevents[0] = self.GRU_Cell_646_SpO2(
                split_single_window_input_data[0] + dbn_transfered_hidden_state_chartevents.narrow(0, 0,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 0, 1))
            hidden_state_chartevents[1] = self.GRU_Cell_778_PaCO2(
                split_single_window_input_data[1] + dbn_transfered_hidden_state_chartevents.narrow(0, 1,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 1, 1))
            hidden_state_chartevents[2] = self.GRU_Cell_780_Arterial_PH(
                split_single_window_input_data[2] + dbn_transfered_hidden_state_chartevents.narrow(0, 2,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 2, 1))
            hidden_state_chartevents[3] = self.GRU_Cell_220045_Heart_Rate(
                split_single_window_input_data[3] + dbn_transfered_hidden_state_chartevents.narrow(0, 3,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 3, 1))
            hidden_state_chartevents[4] = self.GRU_Cell_220050_ABP_mean(
                split_single_window_input_data[4] + dbn_transfered_hidden_state_chartevents.narrow(0, 4,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 4, 1))
            hidden_state_chartevents[5] = self.GRU_Cell_220051_ABP_systolic(
                split_single_window_input_data[5] + dbn_transfered_hidden_state_chartevents.narrow(0, 5,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 5, 1))
            hidden_state_chartevents[6] = self.GRU_Cell_220210_Respiratory_Rate(
                split_single_window_input_data[6] + dbn_transfered_hidden_state_chartevents.narrow(0, 6,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 6, 1))
            hidden_state_chartevents[7] = self.GRU_Cell_220739_coma_scale_eye(
                split_single_window_input_data[7] + dbn_transfered_hidden_state_chartevents.narrow(0, 7,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 7, 1))
            hidden_state_chartevents[8] = self.GRU_Cell_223762_temperature(
                split_single_window_input_data[8] + dbn_transfered_hidden_state_chartevents.narrow(0, 8,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 8, 1))
            hidden_state_chartevents[9] = self.GRU_Cell_223835_fractional_inspired_oxygen(
                split_single_window_input_data[9] + dbn_transfered_hidden_state_chartevents.narrow(0, 9,
                                                                                                   1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 9, 1))
            hidden_state_chartevents[10] = self.GRU_Cell_223900_coma_scale_verbal(
                split_single_window_input_data[10] + dbn_transfered_hidden_state_chartevents.narrow(0, 10,
                                                                                                    1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 10, 1))
            hidden_state_chartevents[11] = self.GRU_Cell_223901_coma_scale_motor(
                split_single_window_input_data[11] + dbn_transfered_hidden_state_chartevents.narrow(0, 11,
                                                                                                    1) + cpt_medication_representation_vectors,
                clone_hidden_state_chartevents.narrow(0, 11, 1))

        hidden_state_labevents = torch.zeros(len(self.labevents_itemids), self.hidden_state_size, dtype=torch.float)
        for i, input_data_single_window in enumerate(input_data_labevents.chunk(input_data_labevents.size(0), dim=0)):
            split_single_window_input_data = torch.split(input_data_single_window, self.input_size, 1)

            cpt_medication_representation_vectors = split_single_window_input_data[11] + split_single_window_input_data[
                12]

            clone_hidden_state_labevents = hidden_state_labevents.clone()
            dbn_transfered_hidden_state_labevents = torch.matmul(self.dbn_matrix_labevents,
                                                                 clone_hidden_state_labevents)
            # [50882, 50885, 50893, 50902, 50912, 50931, 50971, 50983, 51006, 51221, 51301]
            hidden_state_labevents[0] = self.GRU_Cell_50882_Bicarbonate(
                split_single_window_input_data[0] + dbn_transfered_hidden_state_labevents.narrow(0, 0,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 0, 1))
            hidden_state_labevents[1] = self.GRU_Cell_50885_Bilirubin(
                split_single_window_input_data[1] + dbn_transfered_hidden_state_labevents.narrow(0, 1,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 1, 1))
            hidden_state_labevents[2] = self.GRU_Cell_50893_Calcium(
                split_single_window_input_data[2] + dbn_transfered_hidden_state_labevents.narrow(0, 2,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 2, 1))
            hidden_state_labevents[3] = self.GRU_Cell_50902_Chloride(
                split_single_window_input_data[3] + dbn_transfered_hidden_state_labevents.narrow(0, 3,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 3, 1))
            hidden_state_labevents[4] = self.GRU_Cell_50912_Creatinine(
                split_single_window_input_data[4] + dbn_transfered_hidden_state_labevents.narrow(0, 4,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 4, 1))
            hidden_state_labevents[5] = self.GRU_Cell_50931_Glucose(
                split_single_window_input_data[5] + dbn_transfered_hidden_state_labevents.narrow(0, 5,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 5, 1))
            hidden_state_labevents[6] = self.GRU_Cell_50971_Potassium(
                split_single_window_input_data[6] + dbn_transfered_hidden_state_labevents.narrow(0, 6,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 6, 1))
            hidden_state_labevents[7] = self.GRU_Cell_50983_Sodium(
                split_single_window_input_data[7] + dbn_transfered_hidden_state_labevents.narrow(0, 7,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 7, 1))
            hidden_state_labevents[8] = self.GRU_Cell_51006_BUN(
                split_single_window_input_data[8] + dbn_transfered_hidden_state_labevents.narrow(0, 8,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 8, 1))
            hidden_state_labevents[9] = self.GRU_Cell_51221_Hematocrit(
                split_single_window_input_data[9] + dbn_transfered_hidden_state_labevents.narrow(0, 9,
                                                                                                 1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 9, 1))
            hidden_state_labevents[10] = self.GRU_Cell_51301_white_blodd_cell(
                split_single_window_input_data[10] + dbn_transfered_hidden_state_labevents.narrow(0, 10,
                                                                                                  1) + cpt_medication_representation_vectors,
                clone_hidden_state_labevents.narrow(0, 10, 1))

        rnn_output_chartevents = hidden_state_chartevents.narrow(0, 0, 1)
        for i in range(1, len(hidden_state_chartevents)):
            rnn_output_chartevents = torch.cat((rnn_output_chartevents, hidden_state_chartevents.narrow(0, i, 1)), 1)
        rnn_output_labevents = hidden_state_labevents.narrow(0, 0, 1)
        for i in range(1, len(hidden_state_labevents)):
            rnn_output_labevents = torch.cat((rnn_output_labevents, hidden_state_labevents.narrow(0, i, 1)), 1)
        final_output = self.fc(torch.cat((rnn_output_chartevents, rnn_output_labevents), 1))
        return final_output

    def forward(self, anchor_chartevents=None, anchor_labevents=None, positive_chartevents=None,
                positive_labevents=None, negative_chartevents=None, negative_labevents=None):
        output1 = None
        output2 = None
        output3 = None
        if anchor_chartevents is not None and anchor_labevents is not None:
            output1 = self.forward_once(anchor_chartevents, anchor_labevents)
        if positive_chartevents is not None and negative_labevents is not None:
            output2 = self.forward_once(positive_chartevents, positive_labevents)
        if negative_chartevents is not None and negative_labevents is not None:
            output3 = self.forward_once(negative_chartevents, negative_labevents)
        return output1, output2, output3

    def events_to_windows(self, input_file, eventType='chartevents'):

        chartevents_variables = [646, 778, 780, 220045, 220050, 220051, 220210, 220739, 223762, 223835, 223900, 223901]
        labevents_variables = [50882, 50885, 50893, 50902, 50912, 50931, 50971, 50983, 51006, 51221, 51301]
        chartevents_normal_status = {646: [2649], 778: [2782], 780: [2785], 220045: [1280], 220050: [1283],
                                     220051: [1288], 220210: [1293], 220739: [1295], 223762: [1316], 223835: [1319],
                                     223900: [1325], 223901: [1331]}
        labevents_normal_status = {50882: [2061], 50885: [2064], 50893: [2067], 50902: [2070], 50912: [2074],
                                   50931: [2077], 50971: [2081], 50983: [2084], 51006: [2087], 51221: [2223],
                                   51301: [2227]}
        processing_variables = []
        normal_status = {}
        window_number_index = -1

        if eventType == 'chartevents':
            processing_variables = chartevents_variables
            normal_status = chartevents_normal_status
            window_number_index = 5
        else:
            processing_variables = labevents_variables
            normal_status = labevents_normal_status
            window_number_index = 4

        all_patient_event_sequence = {}
        single_patient_event_sequence = []
        current_window_sequence = {}
        previous_window_sequence = normal_status.copy()
        previous_patient = ''
        previous_window_number = -1

        event_data_file = open(input_file)
        event_data_csv_reader = csv.reader(event_data_file)
        for line in event_data_csv_reader:
            subject_id = line[0]
            hadm_id = line[1]
            itemid = self.itemid_filter(line[2])
            event_id = int(line[3])
            current_window_number = int(line[window_number_index])
            current_patient = subject_id + ',' + hadm_id
            if current_patient == previous_patient:  # the same patient
                if current_window_number == previous_window_number:  # the same window number
                    if itemid in current_window_sequence.keys():
                        current_window_sequence[itemid].append(event_id)
                    else:
                        current_window_sequence[itemid] = [event_id]
                else:  # a different window number
                    # missing value imputation
                    for processing_itemid in processing_variables:
                        if processing_itemid not in current_window_sequence.keys():
                            # imputed with the last value in previous window
                            current_window_sequence[processing_itemid] = [
                                previous_window_sequence.get(processing_itemid)[-1]]
                    for i in range(0, current_window_number - previous_window_number):
                        single_patient_event_sequence.append(current_window_sequence.copy())
                    previous_window_number = current_window_number
                    previous_window_sequence = current_window_sequence.copy()
                    current_window_sequence.clear()
                    current_window_sequence[itemid] = [event_id]
            else:  # a different patient
                # handl the last window of the previous patient
                if previous_patient != '':
                    for processing_itemid in processing_variables:
                        if processing_itemid not in current_window_sequence:
                            current_window_sequence[processing_itemid] = [
                                previous_window_sequence.get(processing_itemid)[-1]]
                    single_patient_event_sequence.append(current_window_sequence.copy())
                    all_patient_event_sequence[previous_patient] = single_patient_event_sequence[:]
                    del single_patient_event_sequence[:]
                previous_window_sequence = normal_status.copy()
                current_window_sequence.clear()
                current_window_sequence[itemid] = [event_id]
                previous_patient = current_patient
                previous_window_number = current_window_number

        for processing_itemid in processing_variables:
            if processing_itemid not in current_window_sequence.keys():
                current_window_sequence[processing_itemid] = [previous_window_sequence.get(processing_itemid)[-1]]
        single_patient_event_sequence.append(current_window_sequence.copy())
        all_patient_event_sequence[previous_patient] = single_patient_event_sequence[:]

        return all_patient_event_sequence

    def itemid_filter(self, itemid):
        if ':' in itemid:
            return int(itemid.split(':')[0])
        else:
            if len(itemid) == 5:
                return 200  # cpt-event
            else:
                return 100  # prescription



class ConsrastiveLoss(torch.nn.Module):
    def __init__(self, margin=5.0):
        super(ConsrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, query, positive, negative):
        distance_positive = F.pairwise_distance(query, positive)
        distance_negative = F.pairwise_distance(query, negative)
        distance = torch.sum(torch.pow(distance_positive, 2) - torch.pow(distance_negative, 2)) + self.margin
        dist_hinge = torch.clamp(distance, min=0.0)
        return dist_hinge


class TripletRepresentation:
    def __init__(self, triplet_input_file, code_representation_input_file, patient_event_sequence_input_file):
        if triplet_input_file is not None:
            self.triplets = np.load(triplet_input_file)  # a set
        self.code_representation = np.load(
            code_representation_input_file)  # a dict, key:event_id,value: representation vector
        self.patient_event_sequence = np.load(
            patient_event_sequence_input_file)  # a dict, key:patient_id, value: a list contains dicts for each time_window
        self.event_sequence_chartevents = self.patient_event_sequence['event_sequence_chartevents'].copy()
        self.event_sequence_labevents = self.patient_event_sequence['event_sequence_labevents'].copy()
        self.patient_event_sequence.clear()
        self.chartevents_variables = [646, 778, 780, 220045, 220050, 220051, 220210, 220739, 223762, 223835, 223900,
                                      223901]
        self.labevents_variables = [50882, 50885, 50893, 50902, 50912, 50931, 50971, 50983, 51006, 51221, 51301]
        self.cpt_varialbes = 200
        self.medication_variables = 100
        self.embedding_vector_length = 200

    def get_triplet_representation_random_sampling(self):
        sampled_triplet = random.sample(self.triplets, 1)[0]
        anchort_patient = sampled_triplet[0]
        positive_patient = sampled_triplet[1]
        negative_patient = sampled_triplet[2]
        anchor_chartevents, anchor_labevents = self.get_representation(anchort_patient)
        positive_chartevents, positive_labevents = self.get_representation(positive_patient)
        negative_chartevents, negative_labevents = self.get_representation(negative_patient)
        return anchor_chartevents, anchor_labevents, positive_chartevents, positive_labevents, negative_chartevents, negative_labevents

    def get_representation(self, patient_id):
        chartevents_sequence = self.event_sequence_chartevents.get(patient_id)
        labevents_sequence = self.event_sequence_labevents.get(patient_id)
        return self.event_id_sequence2representation_tensor(
            chartevents_sequence, self.chartevents_variables), self.event_id_sequence2representation_tensor(
            labevents_sequence, self.labevents_variables)

    def event_id_sequence2representation_tensor(self, event_sequence, processing_variables):
        single_patient_representations = np.zeros(1)
        for single_window_event_id in event_sequence:
            single_window_representation = np.zeros(1)
            for variable in processing_variables:
                single_variable_representation = np.zeros(self.embedding_vector_length)
                single_variable_event_id = single_window_event_id.get(variable)
                for event_id in single_variable_event_id:
                    single_variable_representation = single_variable_representation + self.code_representation.get(
                        event_id)
                single_variable_representation = single_variable_representation / len(single_variable_event_id)
                if len(single_window_representation) == 1:
                    single_window_representation = single_variable_representation
                else:
                    # single_window_representation = np.vstack(
                    #     (single_window_representation, single_variable_representation))
                    single_window_representation = np.concatenate(
                        (single_window_representation, single_variable_representation))

            cpt_variable_representation = np.zeros(self.embedding_vector_length)
            if self.cpt_varialbes in single_window_event_id.keys():
                for event_id in single_window_event_id.get(self.cpt_varialbes):
                    cpt_variable_representation = cpt_variable_representation + self.code_representation.get(event_id)
                cpt_variable_representation = cpt_variable_representation / len(
                    single_window_event_id.get(self.cpt_varialbes))
            # single_window_representation = np.vstack((single_window_representation, cpt_variable_representation))
            single_window_representation = np.concatenate((single_window_representation, cpt_variable_representation))

            medication_varialbe_representation = np.zeros(self.embedding_vector_length)
            if self.medication_variables in single_window_event_id.keys():
                for event_id in single_window_event_id.get(self.medication_variables):
                    medication_varialbe_representation = medication_varialbe_representation + self.code_representation.get(
                        event_id)
                medication_varialbe_representation = medication_varialbe_representation / len(
                    single_window_event_id.get(self.medication_variables))
            # single_window_representation = np.vstack((single_window_representation, medication_varialbe_representation))
            single_window_representation = np.concatenate(
                (single_window_representation, medication_varialbe_representation))

            if len(single_patient_representations) == 1:
                single_patient_representations = single_window_representation
            else:
                single_patient_representations = np.vstack(
                    (single_patient_representations, single_window_representation))

        return torch.from_numpy(single_patient_representations).float()


