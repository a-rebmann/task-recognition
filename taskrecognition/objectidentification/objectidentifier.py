import time

import pandas as pd

from taskrecognition.const import ui_object_types, COMMON_FILE_EXTENSIONS, UI_ACTION_WORDS, TERMS_FOR_MISSING
from taskrecognition.util.stringutil import clean_attributes, preprocess_label


def tokenize(word):
    cleaned = preprocess_label(word).split(" ")
    return cleaned


def has_no_numbers(piece_of_ty_inf):
    return not any(char.isdigit() for char in piece_of_ty_inf)


class ObjectIdentifier:

    def __init__(self, task_recognizer, stream=True):
        self.task_recognizer = task_recognizer
        self.objects = {}
        self.objet_type_to_enc = {}
        self.idx_to_obj = {}
        self.known_ne = {}
        self.objects_per_event = {}
        self.event_ids_from_which_objects_were_extracted = {}
        self.prev_task_id = -1
        self.max_time = 0
        self.min_time = 1

    def set_objects(self, task):
        type_to_inst = {}
        type_info = set()
        for idx in range(self.prev_task_id+1, task.id+1):
            type_info.update(self.objects_per_event[idx][0])
            for nouns, inst in self.objects_per_event[idx][1].items():
                if nouns not in type_to_inst:
                    type_to_inst[nouns] = set()
                if len(inst) > 0:
                    type_to_inst[nouns].add(list(inst)[0])
        task.objects = type_to_inst
        task.type_info = type_info
        for type_inf in task.objects:
            if type_inf not in self.objet_type_to_enc:
                self.objet_type_to_enc[type_inf] = len(self.objet_type_to_enc)
        self.idx_to_obj = {value: key for key, value in self.objet_type_to_enc.items()}
        self.objects[task.id] = type_to_inst
        self.prev_task_id = task.id
        return task

    def per_event(self, idx, event):
        #values_clean = (idx, clean_attributes(event[self.task_recognizer.value_attributes[-2:]].values))
        # start measuring time
        start = time.perf_counter()
        context = (idx, clean_attributes(event[self.task_recognizer.context_attributes[2:]].values))
        values_plain = (idx, tuple(att for att in event[self.task_recognizer.value_attributes[-2:]].values if
                                    not pd.isna(att) and att not in TERMS_FOR_MISSING))
        context_value_pairs = (context[0], (set([ot for ot in context[1] ]), set(values_plain[1]))) #+ values_clean[1]
        potential_obj_info = context_value_pairs if len(context_value_pairs[1][0]) > 0 and len(context_value_pairs[1][1]) > 0 else None
        type_info, type_to_inst = self.process_object_info([potential_obj_info])
        # end measuring time
        end = time.perf_counter()
        if end - start > self.max_time:
            self.max_time = end - start
        if end - start < self.max_time:
            self.min_time = end - start
        #print("Object identification took " + str(end - start) + " seconds.")
        self.objects_per_event[idx] = type_info, type_to_inst
        return type_info, type_to_inst

    def identify(self, task_events, task):

        values_clean = [(idx, clean_attributes(event[self.task_recognizer.value_attributes].values)) for idx, event in task_events]
        context = [(idx, clean_attributes(event[self.task_recognizer.context_attributes].values)) for idx, event in task_events]
        # context value pairs maps the event id (idx) to a pair that consists of potential object type
        # information and potential object instance information
        # -> (idx, (potential object type information, potential object instance information))
        values_plain = [(idx, tuple(att for att in event[self.task_recognizer.value_attributes[-2:]].values if
                                    not pd.isna(att) and att not in TERMS_FOR_MISSING)) for idx, event in task_events]
        context_value_pairs = [(context[i][0], (set([ot for ot in context[i][1] + values_clean[i][1]]),
                                                set(values_plain[i][1]))) for i in range(len(task_events))]
        potential_obj_info = [r if len(r[1][0]) > 0 and len(r[1][1]) > 0 else None for r in context_value_pairs]
        type_info, type_to_inst = self.process_object_info(potential_obj_info)
        self.objects[task.id] = type_to_inst
        task.objects = type_to_inst
        task.type_info = type_info
        return task

    def identify_object(self, idx_obj_inf, type_info, type_to_inst):
        if idx_obj_inf is not None:
            idx = idx_obj_inf[0]
            obj_inf = idx_obj_inf[1]
            ty_inf = obj_inf[0]
            inst_inf = obj_inf[1]
            potential_ty_inf = set()
            for piece_of_ty_inf in ty_inf:
                if has_no_numbers(piece_of_ty_inf):
                    preprocessed_and_split = preprocess_label(piece_of_ty_inf).split(" ")
                    if not any(pre in ui_object_types for pre in preprocessed_and_split):
                        potential_ty_inf.add(preprocess_label(piece_of_ty_inf))
            preprocessed_ty_inf = " ".join(potential_ty_inf)
            # Process the text using spaCy
            doc = self.task_recognizer.nlp(preprocessed_ty_inf)
            # Extract nouns, including compound nouns
            nouns = [token.text for token in doc if token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "PROPN" if
                 len(token.text) > 1 and token.text not in UI_ACTION_WORDS and token.text
                 not in COMMON_FILE_EXTENSIONS and token.text not in ui_object_types]
            nes = [ent.text for ent in doc.ents if ent.label_ != "PERSON" or ent.label_ != "ORG" or ent.label_ != "GPE" ]
            nouns = frozenset([noun.replace("id", "") for noun in nouns if noun not in nes])
            compound_nouns = [ent.text for ent in doc.ents if ent.label_ == "NORP" or ent.label_ == "ORG"]
            if len(nouns) > 0:
                type_info.add(nouns)
                for inst in inst_inf:
                    if inst in self.known_ne:
                        contains_ne = self.known_ne[inst]
                    else:
                        # do named entity recognition using spacy
                        doc = self.task_recognizer.nlp(inst)
                        # check if doc contains a named entity of type person or organization or place
                        if any(ent.label_ == "PERSON" or ent.label_ == "ORG" or ent.label_ == "GPE" for ent in
                               doc.ents):
                            contains_ne = True
                        else:
                            contains_ne = False
                        self.known_ne[inst] = contains_ne
                    ###################################################
                    # check for digits, email, url, and named entities
                    ###################################################
                    if inst == "0" or (has_no_numbers(inst) and "@" not in inst
                                       and "http" not in inst and not contains_ne
                                       and not len(inst.split(" ")) >1 and len(inst) < 3):
                        continue
                    if nouns not in type_to_inst:
                        type_to_inst[nouns] = set()
                    type_to_inst[nouns].add(inst)
                type_info.update(compound_nouns)

    def process_object_info(self, potential_obj_info):
        type_info = set()
        type_to_inst = {}
        for idx_obj_inf in potential_obj_info:
            self.identify_object(idx_obj_inf, type_info, type_to_inst)
        return type_info, type_to_inst
