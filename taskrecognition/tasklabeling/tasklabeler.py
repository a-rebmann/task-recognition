from statistics import mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

from taskrecognition.const import COMMON_FILE_EXTENSIONS, TERMS_FOR_MISSING, UI_ACTION_WORDS
from taskrecognition.util.stringutil import clean_attributes


class TaskLabeler:

    def __init__(self, task_recognizer, stream=True):
        self.task_recognizer = task_recognizer
        self.label_matches = {}

    def label(self, task, task_events=None):
        object_terms = set(task.type_info)
        terms_to_check_against = object_terms
        if task_events is not None:
            classes_clean = [(clean_attributes(list(event[self.task_recognizer.context_attributes].values) +
                                               list(event[self.task_recognizer.value_attributes].values))) for
                             idx, event in
                             task_events]
            # split the values inside the tuples
            classes_clean = [item.split() for sublist in classes_clean for item in sublist if len(sublist) > 0]
            # go through list of lists and only keep items of inner lists that have an length of more than 3
            classes_clean = [item for sublist in classes_clean for item in sublist if len(item) > 3 and
                             item not in TERMS_FOR_MISSING and item not in UI_ACTION_WORDS and item
                             not in COMMON_FILE_EXTENSIONS]
            terms_to_check_against = classes_clean
        if self.task_recognizer.labels is not None:
            textual_desc = self.get_label_based_on_max_number_of_exact_matches(terms_to_check_against,
                                                                               self.task_recognizer.labels)  # self.get_label_based_on_objects(set(task.objects.keys()), self.task_recognizer.labels)
            self.label_matches.update({task.id: textual_desc})
            task.label = textual_desc
        return task

    def get_label_based_on_max_number_of_exact_matches(self, objects, labels):
        matched_label = None
        max_matches = {}
        for label in labels:
            matches = 0
            for s in objects:
                # for obj in s:
                if s in label:
                    matches += 1
            max_matches[label] = matches
        if len(max_matches) > 0:
            matched_label = max(max_matches, key=max_matches.get)
        return matched_label

    def get_label_based_on_objects(self, objects, label_list):
        matched_label = None
        max_similarities = {}
        for label in label_list:
            # if any(p in label for e in objects for p in e):
            #     return label
            # else:
            # Process the sentence with spaCy to get the document object
            doc = self.task_recognizer.nlp(label)
            # Compute similarity between the sentence and each type in the list
            similarities = [doc.similarity(self.task_recognizer.nlp(" ".join(s))) for s in objects]
            # Compute the max similarity score
            if len(similarities) > 0:
                mean_similarity = max(similarities)
            else:
                mean_similarity = 0
            max_similarities[label] = mean_similarity
        # match the label with the maximum average similarity
        if len(max_similarities) > 0:
            matched_label = max(max_similarities, key=max_similarities.get)
        return matched_label
