import csv
import itertools
import os
import sys
import time
from collections import Counter
from statistics import mean

import pandas as pd

from eval.label_matching_eval import evaluate_label_matching
from taskrecognition.const import context_attributes_l, value_attributes, context_attributes, semantic_attributes, \
    LABEL, PRED_LABEL
from eval.eval_util import get_edit_distance, get_segment_wise_labels
from eval.object_eval import evaluate_object_detection
from eval.result import Result, get_detailed_csv_header, get_simple_csv_header
from taskrecognition.model.config import Config
from taskrecognition.taskrecognizer import TaskRecognizer
from taskrecognition.util.data_util import prepeare_log
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import rand_score, adjusted_rand_score, jaccard_score, fowlkes_mallows_score

# separators
RESULT_CSV_SEP = ";"

dirname = os.path.dirname(__file__)
path_to_files = os.path.join(dirname, 'data/input')
output_dir = os.path.join(dirname, 'data/output')


lol = {
    "L1.csv": [None],
    "L2.csv": [None],
    "L3.csv": [None],
    "StudentRecord.csv": [None],
    "log3.csv": [None],
    "Reimbursement.csv": [None],
    "agostinelli40.csv": [None],
    "TF_02.csv": [None],
    "TF_03.csv": [None],
    "TF_04.csv": [None],
    "TF_13.csv": [None],
}


def write_results(per_approach_results):
    # write average results
    results_file = output_dir + "/" + "baseline_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=RESULT_CSV_SEP)
        writer.writerow(get_simple_csv_header())
        for approach, reses in per_approach_results.items():
            for res in reses:
                writer.writerow(res.to_simple_csv_line())
    # write per-class results
    results_file = output_dir + "/" + "baseline_results_detailed" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=RESULT_CSV_SEP)
        writer.writerow(get_detailed_csv_header())
        for approach, reses in per_approach_results.items():
            for res in reses:
                for line in res.to_detailed_csv_lines():
                    writer.writerow(line)


def evaluate(task_recognizer, name, config, n=0, m=0, app="Ours"):
    bounds = [task.id for task in task_recognizer.tasks]
    pred = [task.category for task in task_recognizer.tasks]
    lab_to_uis = {lab: [] for lab in set(pred)}
    last_i = 0
    for i, lab in enumerate(pred):
        if i == len(bounds):
            break
        for j in range(last_i, bounds[i]):
            lab_to_uis[lab].append(list(task_recognizer.events.iloc[[j]][LABEL].unique())[0])
        last_i = bounds[i]
    pred_dict = dict()
    clust_to_lab = dict()
    for lab in lab_to_uis.keys():
        cntr = Counter(lab_to_uis[lab])
        mc = cntr.most_common(1)[0][0]
        if mc in pred_dict:
            for elm, count in cntr.items():
                if elm in pred_dict[mc]:
                    pred_dict[mc][elm] += count
                else:
                    pred_dict[mc][elm] = count
        else:
            pred_dict[mc] = dict(cntr)
        clust_to_lab[lab] = mc

    last_i = 0
    task_recognizer.events[PRED_LABEL] = ""
    for i, lab in enumerate(pred):
        if i == len(bounds):
            break
        for j in range(last_i, bounds[i] + 1):
            if j < len(task_recognizer.events):
                task_recognizer.events.loc[j, PRED_LABEL] = clust_to_lab[lab]
        last_i = bounds[i] + 1
    eval_events = task_recognizer.events[task_recognizer.events[PRED_LABEL] != ""]
    ordered_labs = list(eval_events[LABEL].unique())
    w_scores = precision_recall_fscore_support(list(eval_events[LABEL].values), list(eval_events[PRED_LABEL].values),
                                               average='weighted')
    per_class = precision_recall_fscore_support(list(eval_events[LABEL].values), list(eval_events[PRED_LABEL].values),
                                                labels=ordered_labs, average=None)
    rand_mic = rand_score(list(eval_events[LABEL].values), list(eval_events[PRED_LABEL].values))
    ari_mic = adjusted_rand_score(list(eval_events[LABEL].values), list(eval_events[PRED_LABEL].values))
    jaccard_mic = jaccard_score(list(eval_events[LABEL].values), list(eval_events[PRED_LABEL].values),
                                average='weighted')
    fowlkes_mallows_mic = fowlkes_mallows_score(list(eval_events[LABEL].values), list(eval_events[PRED_LABEL].values))
    seg_true_y, seg_pred_y = get_segment_wise_labels(eval_events, bounds, task_recognizer.context_attributes)
    rand_mac = rand_score(seg_true_y, seg_pred_y)
    ari_mac = adjusted_rand_score(seg_true_y, seg_pred_y)
    jaccard_mac = jaccard_score(seg_true_y, seg_pred_y, average='weighted')
    fowlkes_mallows_mac = fowlkes_mallows_score(seg_true_y, seg_pred_y)
    print(rand_mac, rand_mic, jaccard_mac, jaccard_mic)
    df_copy = prepeare_log(name)
    # print(len(df), len(df_2))
    ed, num_pred, num_actual = get_edit_distance(bounds, df_copy, task_recognizer.context_attributes)
    print("ED:", ed)
    result = Result(log=name, approach=app, config=config, with_labels=(len(task_recognizer.labels) > 0),
                    weighted_prec=w_scores[0], weighted_rec=w_scores[1],
                    weighted_f1=w_scores[2],
                    per_class_prec={ordered_labs[i]: per_class[0][i] for i in range(len(per_class[0]))},
                    per_class_rec={ordered_labs[i]: per_class[1][i] for i in range(len(per_class[1]))},
                    per_class_f1={ordered_labs[i]: per_class[2][i] for i in range(len(per_class[2]))},
                    per_class_support={ordered_labs[i]: per_class[3][i] for i in range(len(per_class[3]))},
                    edit_dist=ed, num_pred=num_pred, num_actual=num_actual, rand_mic=rand_mic, ari_mic=ari_mic,
                    jaccard_mic=jaccard_mic, fowlkes_mallows_mic=fowlkes_mallows_mic,
                    rand_mac=rand_mac, ari_mac=ari_mac,
                    jaccard_mac=jaccard_mac, fowlkes_mallows_mac=fowlkes_mallows_mac,
                    runtime=task_recognizer.runtime, part="full", num_classes=len(task_recognizer.event_classes),
                    sizeof_cooc=sys.getsizeof(task_recognizer.co_occurrence_matrix),
                    max_num_events=task_recognizer.max_events_stored,
                    max_buffer_size=task_recognizer.max_buffer_size,
                    size_clust=sys.getsizeof(task_recognizer.task_categorizer.model),
                    size_full=sys.getsizeof(df),
                    avg_resp_seg=mean(task_recognizer.resp_time_seg) if len(task_recognizer.resp_time_seg) > 0 else 0,
                    avg_resp_cat=mean(task_recognizer.resp_time_clust) if len(
                        task_recognizer.resp_time_clust) > 0 else 0)
    if "L" in name and app == "Legacy":
        result.obj_tp, result.obj_fp, result.obj_fn, result.obj_prec, result.obj_rec, result.obj_f1, \
        result.obj_obj_relations_tp, result.obj_obj_relations_fp, result.obj_obj_relations_fn, \
        result.obj_obj_relations_prec, result.obj_obj_relations_rec, result.obj_obj_relations_f1, \
        result.obj_task_relations_tp, result.obj_task_relations_fp, result.obj_task_relations_fn, \
        result.obj_task_relations_prec, result.obj_task_relations_rec, result.obj_task_relations_f1 = \
            evaluate_object_detection(task_recognizer.object_identifier.objects, name, task_recognizer.tasks)
        if len(task_recognizer.labels) > 0:
            result.label_prec, result.label_rec, result.label_f1, result.label_support = evaluate_label_matching(
                task_recognizer.events, task_recognizer.task_labeler.label_matches)
        result.n = n
        result.m = m
    return result


def run(events, config, name, supervision, results, n=0, m=0, bounds=None, app="Ours"):
    context_atts = context_attributes_l if "L3" in filename else context_attributes
    value_atts = value_attributes
    semantic_atts = semantic_attributes
    task_recognizer = TaskRecognizer(config.max_buffer_size, config.warm_up_events, config.similarity_thresh,
                                     context_atts, value_atts, semantic_atts, labels=supervision)
    if bounds:
        task_recognizer.recognize_with_given_bounds(events, bounds)
    else:
        app = "Ours"
        task_recognizer.recognize_from_stream(events)
    print(bounds)
    results[app].append(evaluate(task_recognizer, name, config, n, m, app))


if __name__ == '__main__':
    baseline_segmentation_results = pd.read_pickle("eval/baseline/baseline_segmentation_leno.pkl") #  SET THE PRECOMPUTED SEGMENTAGTION FILE HERE [baseline_perfect_seg_records, baseline_segmentation_prev_and_leno.pkl, baseline_segmentation_urabe.pkl, baseline_segmentation_leno.pkl]
    results = {"Urabe et al. (online)": [], "Urabe et al.": [], "Ours old": [], "Leno et al.": [], "Perfect segmentation": []}
    start = time.time()
    for index, row in baseline_segmentation_results.iterrows():
        filename = row["name"]
        print(filename)
        if ".txt" in filename:
            filename = filename.replace(".txt", "")
        approach = row["approach"]
        if "Leno" in approach:
            approach = "Leno et al."
        df = prepeare_log(filename)
        c = row["config"]
        bounds = row["bounds"]
        print(bounds)
        if bounds is None or len(bounds) == 0:
            print("empty bounds")
            continue
        parts = c.split("-")
        for supervision in lol[filename]:
            print(c, supervision, filename, approach)
            run(df, Config(max_buffer_size=int(parts[3].split("=")[1]),
                           warm_up_events=int(parts[4].split("=")[1]),
                           similarity_thresh=float(parts[2].split("=")[1])),
                filename, supervision, results, n=parts[0].split("=")[1],
                m=parts[1].split("=")[1], bounds=bounds, app=approach)
    print("Total time:", time.time() - start)
    write_results(results)


    # all_approaches = [(row["approach"], row["approach"])  for index, row in baseline_segmentation_results.iterrows()]
    # all_approaches = [a for b in all_approaches for a in b]
    # all_approaches.reverse()
    # print(all_approaches)