import json
import pickle

import pandas as pd


def evaluate_object_detection(objects, name, tasks):
    # read gold standard from pickle file
    with open("./eval/gs/gs_" + name.replace(".csv", "") + ".pickle", 'rb') as f:
        gold_standard = pickle.load(f)
        gold_standard_df = pd.DataFrame(list(gold_standard.items()), columns=['bound', 'objs'])
        objects_df = pd.DataFrame(list(objects.items()), columns=['bound', 'objs'])

        # go through the bounds and actual objects in the gold standard and check if they are in the objects
        # if a bound is in the gold standard but not in objects, get the closest bound in the gold standard to use.
        # if for that bound the actual objects are not in the objects, add them to the false negatives,
        # if they are in the objects, add them to the true positives
        # if there are objects that are not in the gold standard, add them to the false positives
        # also keep track of the number of correct object-to-task and object to object relations
        false_negatives = 0
        true_positives = 0
        false_positives = 0

        tp_task_object_relations = 0
        tp_object_object_relations = 0
        fp_task_object_relations = 0
        fp_object_object_relations = 0
        fn_task_object_relations = 0
        fn_object_object_relations = 0

        for bound, actual_objects in gold_standard.items():
            if bound not in objects:
                # find closest key to bound in objects
                closest_bound = min(objects.keys(), key=lambda x: abs(x - bound))
                for actual_object in actual_objects:
                    if len(actual_objects[actual_object]) == 0:
                        continue
                    pred_objects = objects[closest_bound].keys()
                    if not any(o in actual_object for s in pred_objects for o in s):
                        false_negatives += 1
                        fn_task_object_relations += 1
                        for other_object in actual_objects:
                            if other_object != actual_object:
                                fn_object_object_relations += 1
                    else:
                        true_positives += 1
                        tp_task_object_relations += 1
                        for other_object in actual_objects:
                            if other_object != actual_object:
                                tp_object_object_relations += 1

                # check if there are objects in objects that are not in actual objects
                for obj, inst in objects[closest_bound].items():
                    if len(inst) == 0:
                        continue
                    if not any(s in actual_object for s in obj for o in s for actual_object in actual_objects):
                        false_positives += 1
                        fp_task_object_relations += 1
                        for actual_object in actual_objects:
                            if obj != actual_object:
                                fp_object_object_relations += 1
            else:
                # check if actual objects are in objects
                for actual_object in actual_objects:
                    if len(actual_objects[actual_object]) == 0:
                        continue
                    pred_objects = objects[bound].keys()
                    if not any(o in actual_object for s in pred_objects for o in s):
                        false_negatives += 1
                        fn_task_object_relations += 1
                        for other_object in actual_objects:
                            if other_object != actual_object:
                                fn_object_object_relations += 1

                    else:
                        true_positives += 1
                        tp_task_object_relations += 1
                        for other_object in actual_objects:
                            if other_object != actual_object:
                                tp_object_object_relations += 1
                # check if there are objects in objects that are not in actual objects
                for obj, inst in objects[bound].items():
                    if len(inst) == 0:
                        continue
                    if not any(s in actual_object for s in obj for o in s for actual_object in actual_objects):
                        false_positives += 1
                        fp_task_object_relations += 1
                        for actual_object in actual_objects:
                            if obj != actual_object:
                                fp_object_object_relations += 1

        # calculate precision, recall and f1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        precision_task_object_relations = tp_task_object_relations / (tp_task_object_relations + fp_task_object_relations) if (tp_task_object_relations + fp_task_object_relations) > 0 else 0
        recall_task_object_relations = tp_task_object_relations / (tp_task_object_relations + fn_task_object_relations) if (tp_task_object_relations + fn_task_object_relations) > 0 else 0
        f1_task_object_relations = 2 * ((precision_task_object_relations * recall_task_object_relations) / (precision_task_object_relations + recall_task_object_relations)) if (precision_task_object_relations + recall_task_object_relations) > 0 else 0
        precision_object_object_relations = tp_object_object_relations / (tp_object_object_relations + fp_object_object_relations) if (tp_object_object_relations + fp_object_object_relations) > 0 else 0
        recall_object_object_relations = tp_object_object_relations / (tp_object_object_relations + fn_object_object_relations) if (tp_object_object_relations + fn_object_object_relations) > 0 else 0
        f1_object_object_relations = 2 * ((precision_object_object_relations * recall_object_object_relations) / (precision_object_object_relations + recall_object_object_relations)) if (precision_object_object_relations + recall_object_object_relations) > 0 else 0
        task_relations = find_related_tasks(tasks)
        true_relations = {}
        for bound, actual_objects in gold_standard.items():
            true_relations[bound] = []
            for other_bound, other_actual_objects in gold_standard.items():
                if bound != other_bound:
                    if any(s in actual_objects.items() for s in other_actual_objects.items()) and \
                            any(s in other_actual_objects.items() for s in actual_objects.items()):
                        true_relations[bound].append(other_bound)
        print(precision, recall, f1)
        return true_positives, false_positives, false_negatives, precision, recall, f1, tp_object_object_relations, fp_object_object_relations, fn_object_object_relations, precision_object_object_relations, recall_object_object_relations, f1_object_object_relations, tp_task_object_relations, fp_task_object_relations, fn_task_object_relations, precision_task_object_relations, recall_task_object_relations, f1_task_object_relations


def find_related_tasks(tasks):
    task_relations = {}
    for task in tasks:
        task_relations[task.id] = []
        for other_task in tasks:
            if task.id != other_task.id:
                if any(s in task.objects.items() for s in other_task.objects.items()) and \
                        any(s in other_task.objects.items() for s in task.objects.items()):
                    task_relations[task.id].append(other_task.id)
    return task_relations


def evaluate_objects_per_event(objects_per_event, name):
    with open("./eval/gs/gs_" + name.replace(".csv", "") + ".json", 'rb') as f:
        gold_standard = json.load(f)
        gold_standard = {int(key): value for key, value in gold_standard.items()}

        false_negatives = 0
        fn = dict()
        fp = dict()
        true_positives = 0
        false_positives = 0
        for idx, objs in objects_per_event.items():
            for actual_objs, act_insts in gold_standard[idx].items():
                if actual_objs == '' or len(act_insts) == 0 or act_insts[0] == '':
                    continue
                if len(objs) == 0:
                    false_negatives += 1
                    fn[idx] = actual_objs
                    continue
                if not any(actual_obj in obj.split(" ") for obj in objs for actual_obj in actual_objs.split(" ")):
                    false_negatives += 1
                    fn[idx] = actual_objs
            for obj, insts in objs.items():
                os = obj.split(" ")
                acts = [x.split(" ") for x in gold_standard[idx]]
                acts = [item for sublist in acts for item in sublist]
                if any(o in acts for o in os):
                    true_positives += 1
                else:
                    false_positives += 1
                    fp[idx] = obj
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        print(precision, recall, f1)
        return true_positives, false_positives, false_negatives, precision, recall, f1
