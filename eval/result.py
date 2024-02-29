def get_simple_csv_header():
    return ["approach", "log", "config", "with_labels",
            "precision", "recall", "F$_1$-score", "ED", "Seg. found", "Seg. true",
            "rand_mic", "ari_mic", "jaccard_mic", "fowlkes_mallows_mic", "rand_mac", "ari_mac", "jaccard_mac",
            "fowlkes_mallows_mac",
            "runtime", "part", "num_classes", "sizeof_cooc", "max_num_events",
            "max_buffer_size", "size_clust", "size_full", "avg_resp_seg", "avg_resp_cat",
            "obj_tp", "obj_fp", "obj_fn", "obj_prec", "obj_rec", "obj_f1",
            "obj_obj_relations_tp", "obj_obj_relations_fp", "obj_obj_relations_fn", "obj_obj_relations_prec",
            "obj_obj_relations_rec", "obj_obj_relations_f1",
            "obj_task_relations_tp", "obj_task_relations_fp", "obj_task_relations_fn", "obj_task_relations_prec",
            "obj_task_relations_rec", "obj_task_relations_f1",
            "label_prec", "label_rec", "label_f1", "label_support", "n", "m"]


def get_detailed_csv_header():
    return ["approach", "log", "config", "with_labels",
            "label", "precision", "recall", "F$_1$-score", "support", "ED", "Seg. found",
            "Seg. true",
            "rand_mic", "ari_mic", "jaccard_mic", "fowlkes_mallows_mic", "rand_mac", "ari_mac", "jaccard_mac",
            "fowlkes_mallows_mac",
            "runtime", "part", "num_classes", "sizeof_cooc", "max_num_events",
            "max_buffer_size", "size_clust", "size_full", "avg_resp_seg", "avg_resp_cat",
            "obj_tp", "obj_fp", "obj_fn", "obj_prec", "obj_rec", "obj_f1",
            "obj_obj_relations_tp", "obj_obj_relations_fp", "obj_obj_relations_fn", "obj_obj_relations_prec",
            "obj_obj_relations_rec", "obj_obj_relations_f1",
            "obj_task_relations_tp", "obj_task_relations_fp", "obj_task_relations_fn", "obj_task_relations_prec",
            "obj_task_relations_rec", "obj_task_relations_f1",
            "label_prec", "label_rec", "label_f1", "label_support", "n", "m"]


class Result:

    def __init__(self, approach, log, config, with_labels, weighted_prec, weighted_rec, weighted_f1, per_class_prec, per_class_rec,
                 per_class_f1, per_class_support, edit_dist, num_pred, num_actual, rand_mic, ari_mic, jaccard_mic,
                 fowlkes_mallows_mic, rand_mac, ari_mac, jaccard_mac, fowlkes_mallows_mac, runtime, part,
                 num_classes, sizeof_cooc, max_num_events, max_buffer_size, size_clust, size_full, avg_resp_seg,
                 avg_resp_cat):
        self.approach = approach
        self.log = log
        self.config = config
        self.with_labels = with_labels
        self.weighted_prec = weighted_prec
        self.weighted_rec = weighted_rec
        self.weighted_f1 = weighted_f1
        self.per_class_prec = per_class_prec
        self.per_class_rec = per_class_rec
        self.per_class_f1 = per_class_f1
        self.per_class_support = per_class_support
        self.edit_dist = edit_dist
        self.num_pred = num_pred
        self.num_actual = num_actual
        self.rand_mic = rand_mic
        self.ari_mic = ari_mic
        self.jaccard_mic = jaccard_mic
        self.fowlkes_mallows_mic = fowlkes_mallows_mic
        self.rand_mac = rand_mac
        self.ari_mac = ari_mac
        self.jaccard_mac = jaccard_mac
        self.fowlkes_mallows_mac = fowlkes_mallows_mac

        self.run_time = runtime
        self.part = part
        self.num_classes = num_classes
        self.sizeof_cooc = sizeof_cooc
        self.max_num_events = max_num_events
        self.max_buffer_size = max_buffer_size
        self.size_clust = size_clust
        self.size_full = size_full
        self.avg_resp_seg = avg_resp_seg
        self.avg_resp_cat = avg_resp_cat

        # object scores
        self.obj_tp = 0
        self.obj_fp = 0
        self.obj_fn = 0
        self.obj_prec = 0
        self.obj_rec = 0
        self.obj_f1 = 0
        self.obj_obj_relations_tp = 0
        self.obj_obj_relations_fp = 0
        self.obj_obj_relations_fn = 0
        self.obj_obj_relations_prec = 0
        self.obj_obj_relations_rec = 0
        self.obj_obj_relations_f1 = 0
        self.obj_task_relations_tp = 0
        self.obj_task_relations_fp = 0
        self.obj_task_relations_fn = 0
        self.obj_task_relations_prec = 0
        self.obj_task_relations_rec = 0
        self.obj_task_relations_f1 = 0

        # Label matching scores
        self.label_prec = 0
        self.label_rec = 0
        self.label_f1 = 0
        self.label_support = 0

        self.n = 0
        self.m = 0

        self.partial_results = []

    def to_simple_csv_line(self):
        return [self.approach, self.log, str(self.config), str(self.with_labels), str(self.weighted_prec), str(self.weighted_rec),
                str(self.weighted_f1), str(self.edit_dist), str(self.num_pred), str(self.num_actual),
                str(self.rand_mic), str(self.ari_mic), str(self.jaccard_mic), str(self.fowlkes_mallows_mic),
                str(self.rand_mac), str(self.ari_mac), str(self.jaccard_mac), str(self.fowlkes_mallows_mac),
                str(self.run_time), self.part, self.num_classes, self.sizeof_cooc, self.max_num_events,
                self.max_buffer_size, self.size_clust, self.size_full, self.avg_resp_seg, self.avg_resp_cat,
                self.obj_tp, self.obj_fp, self.obj_fn, self.obj_prec, self.obj_rec, self.obj_f1,
                self.obj_obj_relations_tp, self.obj_obj_relations_fp, self.obj_obj_relations_fn,
                self.obj_obj_relations_prec, self.obj_obj_relations_rec, self.obj_obj_relations_f1,
                self.obj_task_relations_tp, self.obj_task_relations_fp, self.obj_task_relations_fn,
                self.obj_task_relations_prec, self.obj_task_relations_rec, self.obj_task_relations_f1,
                self.label_prec, self.label_rec, self.label_f1, self.label_support, self.n, self.m]

    def to_detailed_csv_lines(self):
        return [[self.approach, self.log, str(self.config), str(self.with_labels), label, str(self.per_class_prec[label]),
                 str(self.per_class_rec[label]), str(self.per_class_f1[label]), str(self.per_class_support[label]),
                 str(self.edit_dist), str(self.num_pred), str(self.num_actual),
                 str(self.rand_mic), str(self.ari_mic), str(self.jaccard_mic), str(self.fowlkes_mallows_mic),
                 str(self.rand_mac), str(self.ari_mac), str(self.jaccard_mac), str(self.fowlkes_mallows_mac),
                 str(self.run_time), self.part, self.num_classes, self.sizeof_cooc, self.max_num_events,
                 self.max_buffer_size, self.size_clust, self.size_full, self.avg_resp_seg, self.avg_resp_cat,
                 self.obj_tp, self.obj_fp, self.obj_fn, self.obj_prec, self.obj_rec, self.obj_f1,
                 self.obj_obj_relations_tp, self.obj_obj_relations_fp, self.obj_obj_relations_fn,
                 self.obj_obj_relations_prec, self.obj_obj_relations_rec, self.obj_obj_relations_f1,
                 self.obj_task_relations_tp, self.obj_task_relations_fp, self.obj_task_relations_fn,
                 self.obj_task_relations_prec, self.obj_task_relations_rec, self.obj_task_relations_f1,
                 self.label_prec, self.label_rec, self.label_f1, self.label_support, self.n, self.m] for label in self.per_class_prec.keys()]
