import sys
import time
from collections import Counter
import numpy as np
import pandas as pd
import spacy

from taskrecognition.const import COMPLETE_INDICATORS, TIMESTAMP, OVERHEAD_INDICATORS, APPLICATION, TERMS_FOR_MISSING, \
    ui_object_types
from taskrecognition.model.task import Task
from taskrecognition.objectidentification.objectidentifier import ObjectIdentifier, has_no_numbers
from taskrecognition.taskcategorization.taskcategorizer import TaskCategorizer
from taskrecognition.tasklabeling.tasklabeler import TaskLabeler
from taskrecognition.util.stringutil import preprocess_label, clean_attributes
from sklearn.metrics.pairwise import cosine_similarity


class TaskRecognizer:

    def __init__(self, max_buffer_size, warm_up_events, similarity_thresh, c_attributes, v_attributes, s_attributes,
                 consider_overhead_tasks_separately=False, min_occurrence_for_df_check=3, labels=None):
        # Parameters
        self.online = False
        self.max_buffer_size = max_buffer_size
        self.warm_up_events = warm_up_events
        self.similarity_thresh = similarity_thresh
        self.context_attributes = c_attributes
        self.value_attributes = v_attributes
        self.semantic_attributes = s_attributes
        self.consider_overhead_tasks_separately = consider_overhead_tasks_separately
        self.min_occurrence_for_df_check = min_occurrence_for_df_check
        self.labels = labels if labels is not None else []

        # State variables
        self.events = pd.DataFrame()
        self.event_classes = list()
        self.event_class_counts = dict()
        self.co_occurrence_matrix = [[0 for _ in range(1000)] for _ in
                                     range(1000)]  # initialize 1000 rows; 1000 columns
        self.current_chunk = list()
        self.buffered_chunks = list()
        self.event_buffer = dict()
        self.tasks = list()
        self.previous_chunk = list()
        self.similarity_scores = list()
        self.directly_follows = dict()
        self.event_counter = 0
        self.warm_up_vectors = []

        # Models
        self.task_categorizer = TaskCategorizer(self)
        self.object_identifier = ObjectIdentifier(self)
        self.task_labeler = TaskLabeler(self)
        self.nlp = spacy.load('en_core_web_md')

        # Metrics
        self.runtime = 0
        self.resp_time_clust = []
        self.resp_time_seg = []
        self.max_events_stored = 0
        self.max_buffer_size = 0

    def completes_chunk(self, semantic_attributes):
        return (any(preprocess_label(str(semantic_attribute)) in COMPLETE_INDICATORS or (
                len(preprocess_label(str(semantic_attribute)).split(" ")) == 2 and
                (preprocess_label(str(semantic_attribute)).split(" ")[0] in COMPLETE_INDICATORS or
                 preprocess_label(str(semantic_attribute)).split(" ")[1] in COMPLETE_INDICATORS))
                    or (len(preprocess_label(str(semantic_attribute)).split(" ")) == 3 and
                        preprocess_label(str(semantic_attribute)).split(" ")[0] in COMPLETE_INDICATORS)
                    for semantic_attribute in semantic_attributes))

    def update_co_occurrence_matrix(self, current_chunk):
        for (index, row) in current_chunk:
            event_class = self.get_event_class(row)
            event_class_index = self.event_classes.index(event_class)
            for (other_index, other_row) in current_chunk:
                other_event_class = self.get_event_class(other_row)
                other_event_class_index = self.event_classes.index(other_event_class)
                self.co_occurrence_matrix[event_class_index][other_event_class_index] += 1

    def compute_contextual_similarity(self, chunk_events_1, chunk_events_2):
        cooc_vecs_1 = [self.co_occurrence_matrix[self.event_classes.index(self.get_event_class(curr[1]))] for curr in
                       chunk_events_1]
        cooc_vecs_2 = [self.co_occurrence_matrix[self.event_classes.index(self.get_event_class(curr[1]))] for curr in
                       chunk_events_2]
        vec_1 = [0 for _ in range(len(self.event_classes))]
        vec_2 = [0 for _ in range(len(self.event_classes))]
        for curr in chunk_events_1:
            vec_1[self.event_classes.index(self.get_event_class(curr[1]))] += 1
        for curr in chunk_events_2:
            vec_2[self.event_classes.index(self.get_event_class(curr[1]))] += 1
        c_1 = np.mean(cooc_vecs_1, axis=0).reshape(1, -1)
        c_2 = np.mean(cooc_vecs_2, axis=0).reshape(1, -1)
        return cosine_similarity([vec_1], [vec_2])[0][0]

    def check_contextually_unrelated(self, potentially_last, potentially_first):
        sim = self.compute_contextual_similarity(potentially_last, potentially_first)
        previous = self.similarity_scores[-1] if len(self.similarity_scores) > 0 else 0
        self.similarity_scores.append(sim)
        # print("Contextual similarity delta: " + str(abs(previous - sim)),
        #      " Contextually unrelated:" + str(abs(previous - sim) < self.similarity_thresh), " Similarity: " + str(sim))
        print("Contextual similarity: " + str(sim), " Contextually unrelated:" + str(sim < self.similarity_thresh))
        return sim < self.similarity_thresh  # abs(previous - sim) < self.similarity_thresh or

    def check_non_overlapping_values(self, potentially_last, potentially_first):
        semantic_attributes_last = [x for event in potentially_last[-2:] for x in
                                    clean_attributes(event[1][self.semantic_attributes].values)]
        semantic_attributes_first = [x for event in potentially_first[:3] for x in
                                     clean_attributes(event[1][self.semantic_attributes].values)]
        values_last = [x for event in potentially_last[-2:] for x in
                       clean_attributes(event[1][self.value_attributes].values)]
        values_first = [x for event in potentially_first[:2] for x in
                        clean_attributes(event[1][self.value_attributes].values)]
        if any(first == last for first in semantic_attributes_first for last in semantic_attributes_last):
            print("non-overlapping values 1: False")
            return False
        # if any(first in COMPLETE_INDICATORS for first in semantic_attributes_first):
        #     print("non-overlapping values 2: False")
        #     return False
        if APPLICATION not in self.context_attributes and any(
                first == last for first in values_first for last in values_last):
            print("non-overlapping values 3: False")
            return False
        if not len(self.semantic_attributes) == 1 and any(
                p_r in po or p_o in pr for pr in semantic_attributes_first for p_r in pr.split(" ") for po in
                semantic_attributes_last for p_o in po.split(" ") if len(p_r) > 2 and len(p_o) > 2 and p_r not in
                                                                     COMPLETE_INDICATORS and p_o not in COMPLETE_INDICATORS
                                                                     and p_o not in ui_object_types):
            print("non-overlapping values 4: False")
            return False
        print("non-overlapping values: True")
        return True

    def no_overhead_task(self, potentially_last):
        over_head = any(preprocess_label(str(semantic_attribute)) in OVERHEAD_INDICATORS for semantic_attribute in
                        [tuple(event[1][self.semantic_attributes].values) for event in potentially_last])
        print("is no overhead task: " + str(not over_head))
        return not over_head

    def check_directly_follows_determinism(self, potentially_last, potentially_first):
        global_class_count = self.event_class_counts[self.get_event_class(potentially_last[-1][1])]
        last_event_class = self.get_event_class(potentially_last[-1][1])
        first_event_class = self.get_event_class(potentially_first[0][1])
        second_last_event_class = self.get_event_class(potentially_first[1][1])
        third_last_event_class = self.get_event_class(potentially_first[2][1])
        df_1, df_2, df_3 = 0, 0, 0
        if (last_event_class, first_event_class) in self.directly_follows[2]:
            df_1 = self.directly_follows[2][(last_event_class, first_event_class)]
        if (last_event_class, second_last_event_class) in self.directly_follows[2]:
            df_2 = self.directly_follows[2][(last_event_class, second_last_event_class)]
        if (last_event_class, third_last_event_class) in self.directly_follows[2]:
            df_3 = self.directly_follows[2][(last_event_class, third_last_event_class)]
        if df_1 > 0 or df_2 > 0 or df_3 > 0:
            df_2 += df_3 + df_1
            if global_class_count <= df_2 and df_2 > self.min_occurrence_for_df_check:
                print("not directly follows deterministic: False")
                return False
        else:
            if global_class_count <= 1.1 * df_2 and df_2 > 4:
                print("not directly follows deterministic: False")
                return False
        print("not directly follows deterministic: True")
        return True

    def not_only_common_instances(self, potentially_last, potentially_first):
        objects_last = [list(list(self.object_identifier.objects_per_event[j[0]][1].values())[0])[0] for j in
                        potentially_last if len(self.object_identifier.objects_per_event[j[0]][1].values()) > 0 and len(
                list(list(self.object_identifier.objects_per_event[j[0]][1].values())[0])) > 0]
        objects_first = [list(list(self.object_identifier.objects_per_event[j[0]][1].values())[0])[0] for j in
                         potentially_first if
                         len(self.object_identifier.objects_per_event[j[0]][1].values()) > 0 and len(
                             list(list(self.object_identifier.objects_per_event[j[0]][1].values())[0])) > 0]
        objects_last = [inst for inst in objects_last if not has_no_numbers(inst) and "TF" not in inst]
        objects_first = [inst for inst in objects_first if not has_no_numbers(inst) and "TF" not in inst]
        # for i in potentially_last:
        #     for j in potentially_first:
        #         if self.object_identifier.objects_per_event[i[0]][1] != self.object_identifier.objects_per_event[j[0]][1] and len(self.object_identifier.objects_per_event[j[0]][1]) > 0:
        #             return True
        res = objects_last != objects_first or len(objects_last) == 0 or len(objects_last) == 0
        return res

    def ends_task(self, potentially_last, potentially_first):
        print("ends task for " + str(potentially_last[-1][0]) + "?")
        main_tic = time.perf_counter()
        res = len(potentially_first) >= 3 and \
              (self.check_contextually_unrelated(potentially_last, potentially_first)
               or self.check_directly_follows_determinism(potentially_last, potentially_first)) and \
              self.check_non_overlapping_values(potentially_last, potentially_first) and \
              (self.consider_overhead_tasks_separately or self.no_overhead_task(potentially_last)) and \
              self.not_only_common_instances(potentially_last, potentially_first)

        # UNCOMMENT ONLY FOR ABLATION STUDY (WITHOUT OBJECT PERSPECTIVE)
        # res = len(potentially_first) >= 3 and \
        #       (self.check_contextually_unrelated(potentially_last, potentially_first)
        #        or self.check_directly_follows_determinism(potentially_last, potentially_first)) and \
        #       self.check_non_overlapping_values(potentially_last, potentially_first) and \
        #       (self.consider_overhead_tasks_separately or self.no_overhead_task(potentially_last))

        # UNCOMMENT ONLY FOR ABLATION STUDY (WITHOUT DATA PERSPECTIVE)
        # res = len(potentially_first) >= 3 and \
        #       (self.check_contextually_unrelated(potentially_last, potentially_first)
        #        or self.check_directly_follows_determinism(potentially_last, potentially_first)) and \
        #       (self.consider_overhead_tasks_separately or self.no_overhead_task(potentially_last)) and \
        #       self.not_only_common_instances(potentially_last, potentially_first)

        # UNCOMMENT ONLY FOR ABLATION STUDY (WITHOUT SEMANTIC PERSPECTIVE)
        # res = len(potentially_first) >= 3 and \
        #       (self.check_contextually_unrelated(potentially_last, potentially_first)
        #        or self.check_directly_follows_determinism(potentially_last, potentially_first)) and \
        #       self.check_non_overlapping_values(potentially_last, potentially_first) and \
        #       self.not_only_common_instances(potentially_last, potentially_first)

        # res = len(potentially_first) >= 3 and \
        #       (self.check_contextually_unrelated(potentially_last, potentially_first)
        #        or self.check_directly_follows_determinism(potentially_last, potentially_first))

        main_toc = time.perf_counter()
        runtime = main_toc - main_tic
        self.resp_time_seg.append(runtime)
        return res

    def prepare_feature_vector(self, task_events, task=None):
        task_event_classes = [self.get_event_class(event) for idx, event in task_events]
        # object_vector = []
        # if task:
        #     object_vector = [1 if self.object_identifier.idx_to_obj[i] in task.objects else 0 for i in range(len(self.object_identifier.idx_to_obj))]
        #     if len(object_vector) < 1000:
        #         object_vector = object_vector + [0 for _ in range(1001 - len(self.object_identifier.idx_to_obj))]
        vector = []
        counts = Counter(task_event_classes)
        event_types = set()
        for event_class in self.event_classes:
            # event_types.add(event_class)
            if event_class in counts:
                vector.append(counts[event_class])
            else:
                vector.append(0)
       # vector = object_vector + vector
        if len(vector) < 1000:
            vector = [len(event_types)] + vector + [0 for _ in
                                                    range(1001 - len(self.event_classes))]  # , len(task.objects)
        return vector

    def _update_directly_follows(self, event_class, index):
        for i in range(1, 10):
            curr = index - i
            if curr in self.event_buffer:
                previous_event_class = self.get_event_class(self.event_buffer[curr])
                if i not in self.directly_follows:
                    self.directly_follows[i] = {}
                if (previous_event_class, event_class) not in self.directly_follows[i]:
                    self.directly_follows[i][(previous_event_class, event_class)] = 0
                self.directly_follows[i][(previous_event_class, event_class)] += 1

    def recognize_from_stream(self, stream):
        self.online = True
        self.events = stream
        stream_tic = time.perf_counter()
        # The index needs to be an integer starting from 0 and increasing by 1 for each event!
        for index, row in self.events.iterrows():
            # add event to the buffer
            self.event_buffer[index] = row
            self.event_counter += 1
            # update event class index
            event_class = self.get_event_class(row)
            if event_class not in self.event_classes:
                self.event_classes.append(event_class)
                self.event_class_counts[event_class] = 0
            self.event_class_counts[event_class] += 1
            # update directly_follows_counts
            if len(self.event_buffer) > 0:
                self._update_directly_follows(event_class, index)
            self.current_chunk.append((index, row))
            semantic_attributes = tuple(row[self.semantic_attributes].values)

            #  IDENTIFY OBJECT INSTANCES
            self.object_identifier.per_event(index, row)

            # check if the chunk is complete
            if self.completes_chunk(semantic_attributes):
                self.update_co_occurrence_matrix(self.current_chunk)
                self.buffered_chunks.append(self.current_chunk)
                self.current_chunk = list()
                if len(self.buffered_chunks) >= 2:
                    potentially_last, potentially_first = self.buffered_chunks[-2], self.buffered_chunks[-1]
                    # check if task ends
                    if self.ends_task(potentially_last, potentially_first):
                        self.create_task_and_clear_buffer(potentially_last[-1][0])
        self.create_task_and_clear_buffer(self.buffered_chunks[-1][-1][0])
        stream_toc = time.perf_counter()
        self.runtime = stream_toc - stream_tic

    def recognize_with_given_bounds(self, stream, bounds):
        self.online = True
        self.events = stream
        stream_tic = time.perf_counter()
        # The index needs to be an integer starting from 0 and increasing by 1 for each event!
        for index, row in self.events.iterrows():
            # add event to the buffer
            self.event_buffer[index] = row
            self.event_counter += 1
            # update event class index
            event_class = self.get_event_class(row)
            if event_class not in self.event_classes:
                self.event_classes.append(event_class)
                self.event_class_counts[event_class] = 0
            self.event_class_counts[event_class] += 1
            # update directly_follows_counts
            if len(self.event_buffer) > 0:
                self._update_directly_follows(event_class, index)
            self.current_chunk.append((index, row))
            if index in bounds:
                self.create_task_and_clear_buffer(index)
            stream_toc = time.perf_counter()
            self.runtime = stream_toc - stream_tic

    def create_task_and_clear_buffer(self, last_idx):
        task = Task()
        # set the index of the last event of the task as the id of the task
        task.id = last_idx
        task_events = [(idx, event) for idx, event in self.event_buffer.items() if idx <= task.id]
        task.start_timestamp = task_events[0][1][TIMESTAMP]
        task.end_timestamp = task_events[-1][1][TIMESTAMP]
        try:
            task = self.object_identifier.set_objects(task)
        except KeyError:
            pass
        # categorization of the task with warm-up events
        vector = self.prepare_feature_vector(task_events, task)
        if self.event_counter >= self.warm_up_events:
            if len(self.warm_up_vectors) > 0:
                self.tasks = self.task_categorizer.categorize(self.warm_up_vectors, self.tasks)
                self.warm_up_vectors = []
            task = self.task_categorizer.categorize([vector], [task])[0]
        else:
            self.warm_up_vectors.append(vector)
        # task = self.object_identifier.identify(task_events, task)
        task = self.task_labeler.label(task, task_events)
        self.tasks.append(task)
        # delete the events of the task from the buffer after counting the events in the buffer
        if len(self.event_buffer) > self.max_events_stored:
            self.max_events_stored = len(self.event_buffer)
        if sys.getsizeof(self.event_buffer) > self.max_buffer_size:
            self.max_buffer_size = sys.getsizeof(self.event_buffer)
        for idx in range(min(self.event_buffer.keys()), task.id + 1):
            del self.event_buffer[idx]

    def recognize_from_log(self, log):
        self.online = False
        self.events = log
        self.warm_up_events = len(self.events)
        self.recognize_from_stream(self.events)

    def get_event_class(self, row):
        return tuple(
            val for val in row[self.context_attributes].values if not pd.isna(val) and val not in TERMS_FOR_MISSING)
