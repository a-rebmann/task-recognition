import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from taskrecognition.const import ALPHABET
from taskrecognition.taskcategorization.clusterer.denstream import DenStream


class TaskCategorizer:

    def __init__(self, task_recognizer, stream=True):
        self.task_recognizer = task_recognizer
        if stream:
            self.model = DenStream(lambd=0.01, eps=1, beta=0.5, mu=3)
        else:
            self.model = None # TODO: add model for offline processing
        self.cluster_explanation = {}
        self.X_train_tf_idf = None
        self.tf_idf_vect = None
        self.terms = None
        self.current_defining_terms = {}
        self.lab_to_type_lab = {}

    def categorize(self, vectors, tasks):
        main_tic = time.perf_counter()
        self.model.partial_fit(vectors)
        categories = self.model.fit_predict(vectors, num_current_features=len(self.task_recognizer.event_classes))
        main_toc = time.perf_counter()
        runtime = main_toc - main_tic
        self.task_recognizer.resp_time_clust.append(runtime)
        for i, category in enumerate(categories):
            if category not in self.cluster_explanation:
                self.lab_to_type_lab[category] = ALPHABET[len(self.lab_to_type_lab) % 26]
                self.cluster_explanation[category] = []
            self.cluster_explanation[category].extend([obj for objs in tasks[i].objects.keys() for obj in objs])
        self.compute_tf_idf()
        for i, task in enumerate(tasks):
            task.category = categories[i]
        return tasks

    def compute_tf_idf(self):
        # transform the tf idf vectorizer
        self.tf_idf_vect = TfidfVectorizer()
        try:
            self.tf_idf_vect.fit([" ".join(self.cluster_explanation[lab]) for lab in self.cluster_explanation.keys()])
            self.terms = self.tf_idf_vect.get_feature_names_out()
        except ValueError:
            #print("No terms found for tf-idf")
            return
        per_clust = {lab: ["expl"] for lab in self.cluster_explanation.keys()}
        currentidx = 0
        done = False
        while any(per_clust[another_cluster][-1] == per_clust[one_cluster][-1] for one_cluster in per_clust.keys() for
                  another_cluster in per_clust.keys() if one_cluster != another_cluster) and not done:
            for lab in self.cluster_explanation.keys():
                feature_array = np.array(self.terms)
                response = self.tf_idf_vect.transform([" ".join(self.cluster_explanation[lab])])
                tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
                if len(tfidf_sorting) > currentidx:
                    top_1 = feature_array[tfidf_sorting][currentidx]
                    per_clust[lab].append(top_1)
                else:
                    done = True
            currentidx += 1
        self.current_defining_terms = per_clust

    def get_defining_terms(self, labels):
        explanations = []
        for lab in labels:
            explanations.append(self.current_defining_terms[lab])
        return explanations

