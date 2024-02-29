from statistics import mean

import editdistance

from taskrecognition.const import LABEL, INDEX, PRED_LABEL, TERMS_FOR_MISSING


def create_tup(context_atts, row, embedding=False):
    tup = []
    for att in context_atts:
        if row[att] not in TERMS_FOR_MISSING:
            tup.append(str(row[att]))
    return tup if embedding else " ".join(tup)


def create_ground_truth_array(df):
    df["GT_SEG"] = df[LABEL] + df[INDEX].astype(str)
    gt_array = []
    prev_label = None
    prev_index = None
    for index, row in df.iterrows():
        if prev_label is not None and row["GT_SEG"] != prev_label:
            gt_array.append(prev_index)
        prev_label = row["GT_SEG"]
        prev_index = index
    gt_array.append(len(df)-1)
    return gt_array


def get_all_uis(df, context_atts):
    all_uis = []
    for index, row in df.iterrows():
        all_uis.append(create_tup(context_atts, row))
    return all_uis


def get_edit_distance(bounds, df, context_atts):
    all_uis = get_all_uis(df, context_atts)
    gt_bounds = create_ground_truth_array(df)
    print("bounds: \n", bounds)
    print(gt_bounds)
    editDistances = []
    curr_start = 0
    discoveredSegments = []
    for curr_bound in bounds:
        pred_idxs = [i for i in range(curr_start, curr_bound+1)]
        discoveredSegments.append(pred_idxs)
        curr_start = curr_bound + 1
    curr_start = 0
    trueSegments = []
    for gt_bound in gt_bounds:
        gt_idxs = [i for i in range(curr_start, gt_bound+1)]
        trueSegments.append(gt_idxs)
        curr_start = gt_bound + 1
    for discovered_seg in discoveredSegments:
        coveredTraces = []
        for true_seg in trueSegments:
            if any([i in discovered_seg for i in true_seg]) and len(true_seg) > 0:
                coveredTraces.append(true_seg)
        editDistance = 1
        min_seg = []
        for true_seg in coveredTraces:
            dist = (editdistance.eval([all_uis[i] for i in discovered_seg if i < len(all_uis)], [all_uis[i] for i in true_seg if i < len(all_uis)]) / max(len(discovered_seg), len(true_seg)))
            if dist < editDistance:
                editDistance = dist
                min_seg = true_seg
        if len(min_seg) > 0:
            editDistances.append(editDistance)
    meanEditDistance = mean(editDistances)
    return meanEditDistance, len(bounds), len(gt_bounds)


def get_segment_wise_labels(df, bounds, context_atts, orig_df=None):
    if orig_df is None:
        orig_df = df
    all_uis = get_all_uis(df, context_atts)
    gt_bounds = create_ground_truth_array(df)
    true_y = []
    pred_y = []
    curr_start = 0
    discoveredSegments = []
    for curr_bound in bounds:
        pred_idxs = [i for i in range(curr_start, curr_bound + 1)]
        discoveredSegments.append(pred_idxs)
        curr_start = curr_bound + 1
    curr_start = 0
    trueSegments = []
    for gt_bound in gt_bounds:
        gt_idxs = [i for i in range(curr_start, gt_bound + 1)]
        trueSegments.append(gt_idxs)
        curr_start = gt_bound + 1
    for discovered_seg in discoveredSegments:
        coveredTraces = []
        for true_seg in trueSegments:
            if any([i in discovered_seg for i in true_seg]) and len(true_seg) > 0:
                coveredTraces.append(true_seg)
        editDistance = 1
        min_seg = []
        for true_seg in coveredTraces:
            dist = (editdistance.eval([all_uis[i] for i in discovered_seg if i < len(all_uis)],
                                      [all_uis[i] for i in true_seg if i < len(all_uis)]) / max(len(discovered_seg),
                                                                                                len(true_seg)))
            if dist < editDistance:
                editDistance = dist
                min_seg = true_seg
        if len(min_seg) > 0:
            true_y.append(orig_df.iloc[min_seg[int(len(min_seg)/2)]][LABEL])
            pred_y.append(orig_df.iloc[discovered_seg[int(len(discovered_seg) / 2)]][PRED_LABEL])
    return true_y, pred_y
