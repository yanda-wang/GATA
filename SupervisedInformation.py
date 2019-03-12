import csv
import numpy as np
import itertools
import math
import seaborn as sns
import matplotlib.pyplot as plt
import operator
import cPickle
from numpy.random import choice


class PatientSimilarityDistance:

    def __init__(self):
        self.patient_number = {}

    def read_patient_icd_code(self, icd_code_input_file):
        patient_icd_code_file = open(icd_code_input_file)
        patient_icd_code_csv_reader = csv.reader(patient_icd_code_file)
        last_key = ''
        icd_code_set = set()
        patient_icd_codes = {}
        self.patient_number.clear()
        patient_count = 0

        for line in patient_icd_code_csv_reader:
            subject_id = line[0]
            hadm_id = line[1]
            current_icd_code = line[2]
            current_key = subject_id + ',' + hadm_id
            if current_key == last_key:  # the same patient
                icd_code_set.add(current_icd_code)
            else:
                if len(icd_code_set) > 0:
                    patient_icd_codes[last_key] = icd_code_set
                    self.patient_number[last_key] = patient_count
                    patient_count = patient_count + 1
                icd_code_set = set()
                icd_code_set.add(current_icd_code)
            last_key = current_key

        if len(icd_code_set) > 0:
            patient_icd_codes[last_key] = icd_code_set
            self.patient_number[last_key] = patient_count

        patient_icd_code_file.close()
        print 'read icd code successfully'
        return patient_icd_codes

    def compute_patient_similarity(self, icd_code_input_file):
        patient_icd_codes = self.read_patient_icd_code(icd_code_input_file)
        patient_size = len(patient_icd_codes)
        patient_similarity_matrix = np.zeros((patient_size, patient_size))
        for pairwise_keys in itertools.combinations_with_replacement(patient_icd_codes.keys(), 2):
            patient_a = pairwise_keys[0]
            patient_b = pairwise_keys[1]
            patient_icd_codes_a = patient_icd_codes.get(patient_a)
            patient_icd_codes_b = patient_icd_codes.get(patient_b)
            patient_similarity = self.patient_similarity_by_icd_code(patient_icd_codes_a, patient_icd_codes_b)
            patient_similarity_matrix[self.patient_number.get(patient_a)][
                self.patient_number.get(patient_b)] = patient_similarity
            patient_similarity_matrix[self.patient_number.get(patient_b)][self.patient_number.get(patient_a)] = \
                patient_similarity_matrix[self.patient_number.get(patient_a)][self.patient_number.get(patient_b)]

        print 'compute similarity matrix successfully'

        return patient_similarity_matrix


    def patient_similarity_by_icd_code(self, patient_a_icd_code_set, patient_b_icd_code_set):
        relevance_score = 0
        for a_icd_code in patient_a_icd_code_set:
            for b_icd_code in patient_b_icd_code_set:
                relevance_score = relevance_score + self.patient_similarity(a_icd_code, b_icd_code)
        relevance_score = float(relevance_score) / float(len(patient_a_icd_code_set) * len(patient_b_icd_code_set))
        return relevance_score

    def patient_similarity(self, icd_code_a, icd_code_b):
        for i in range(0, min(len(icd_code_a), len(icd_code_b))):
            if icd_code_a[i] != icd_code_b[i]:
                return i
        return min(len(icd_code_a), len(icd_code_b))

    
class TripletGenerator:

    def generate_triplets(self, similarity_matirx_input_file, triplets_output_file, sample_triplet_number=1000000,
                          margin_threshold=0.2, max_sampling_iteration=10000000, single_sampling_iteration=1000):
        each_patient_sum_similarity, each_patient_positive_negative_candidates, similarity_matrix, patient_index = self.calculate_probability(
            similarity_matirx_input_file)
        sample_triplet_count = 0
        successfully_sample_count = 0
        triplets_results = set()

        while successfully_sample_count < sample_triplet_number and sample_triplet_count < max_sampling_iteration:
            sample_triplet_count += 1
            anchor_patient = choice(each_patient_sum_similarity.keys(), 1, p=each_patient_sum_similarity.values())[0]
            positive_negative_candidates = each_patient_positive_negative_candidates[anchor_patient]
            positive_patient = choice(positive_negative_candidates.keys(), 1, p=positive_negative_candidates.values())[
                0]
            negative_patient = None
            current_sample_count = 0
            while current_sample_count < single_sampling_iteration:
                current_sample_count += 1
                negative_uniformly_sample = np.random.choice([True, False])
                if negative_uniformly_sample:
                    negative_patient = choice(positive_negative_candidates.keys(), 1)[0]
                else:
                    negative_patient = \
                        choice(positive_negative_candidates.keys(), 1, p=positive_negative_candidates.values())[0]

                similarity_between_anchor_positive = similarity_matrix[patient_index.get(anchor_patient)][
                    patient_index.get(positive_patient)]
                similarity_between_anchor_negative = similarity_matrix[patient_index.get(anchor_patient)][
                    patient_index.get(negative_patient)]
                similarity_difference = similarity_between_anchor_positive - similarity_between_anchor_negative
                if similarity_difference > margin_threshold:
                    break
                else:
                    negative_patient = None

            if negative_patient:
                successfully_sample_count += 1
                current_triplets = [anchor_patient, positive_patient, negative_patient,
                                    similarity_between_anchor_positive, similarity_between_anchor_negative]
                triplets_results.add(tuple(current_triplets))

            if sample_triplet_count % 1000 == 0:
                print sample_triplet_count, len(triplets_results)

        cPickle.dump(triplets_results, open(triplets_output_file, 'wb'))

    def calculate_probability(self, similarity_matrix_input_file):
        reload_objects_simlarity = np.load(similarity_matrix_input_file)
        similarity_matrix = reload_objects_simlarity['similarity_matrix']
        patient_index = reload_objects_simlarity['patient_key2index']
        each_patient_sum_similarity = {}  # key:subject_id+hadm_id,value: sum of similarity
        sum_similarity = np.sum(similarity_matrix, axis=1)
        for (patient, index) in patient_index.items():
            each_patient_sum_similarity[patient] = sum_similarity[index] - similarity_matrix[index, index]

        sorted_each_patient_sum_similarity = sorted(each_patient_sum_similarity.items(), key=operator.itemgetter(1))
        patient_count = len(sorted_each_patient_sum_similarity)
        patient_sampled_as_anchor_probabilit = {}
        for i in range(0, patient_count):
            patient_sampled_as_anchor_probabilit[sorted_each_patient_sum_similarity[i][0]] = 2.0 * (i + 1) / (
                    patient_count * (patient_count + 1))

        each_patient_positive_negative_candidates = {}
        for current_patient in patient_index.keys():
            positive_negative_candidates = {}
            similarity2other_patient = np.copy(similarity_matrix[patient_index[current_patient], :])
            for (key, value) in patient_index.items():
                positive_negative_candidates[key] = similarity2other_patient[value]
            positive_negative_candidates.pop(current_patient, None)

            sorted_candidates = sorted(positive_negative_candidates.items(), key=operator.itemgetter(1))
            candidate_count = len(sorted_candidates)
            patient_sampled_as_positive_negative_probablity = {}
            for i in range(0, candidate_count):
                patient_sampled_as_positive_negative_probablity[sorted_candidates[i][0]] = 2.0 * (i + 1) / (
                        candidate_count * (candidate_count + 1))
            each_patient_positive_negative_candidates[current_patient] = patient_sampled_as_positive_negative_probablity

        return patient_sampled_as_anchor_probabilit, each_patient_positive_negative_candidates, similarity_matrix, patient_index
