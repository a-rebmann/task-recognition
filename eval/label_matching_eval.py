from eval.eval_util import create_ground_truth_array

correct_matching = {
    "Reimbursement": "enter reimbursement info",
    "CopyPasteStuff": "copy area info",
    "StudentRecord": "enter student info",
    "TravelRequest": "enter travel request info",
    "TF_02.csv": "Create structural unit",
    "TF_03.csv": "Create chapter",
    "TF_04.csv": "Create organizational unit",
    "TF_13.csv": "Create specification",
}


def evaluate_label_matching(events, label_matches):
    true_bounds = create_ground_truth_array(events)
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []
    correct_labels = {
        bound: correct_matching[events.iloc[bound]["Task"]] for bound in true_bounds
    }
    for bound, true_label in correct_labels.items():
        if bound not in label_matches:
            # find closest key to bound in objects
            closest_bound = min(label_matches.keys(), key=lambda x: abs(x - bound))
            pred_label = label_matches[closest_bound]
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            if label_matches[closest_bound] == true_label:
                correct += 1
            total += 1
        else:
            pred_label = label_matches[bound]
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            if label_matches[bound] == true_label:
                correct += 1
            total += 1
    # calculate precision, recall and f1 based on true labels and predicted labels with sklearn
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average="weighted")

    return precision, recall, f1, support if support is not None else len(true_labels)

