class Config:

    def __init__(self,  max_buffer_size, warm_up_events, similarity_thresh):
        self.max_buffer_size = max_buffer_size
        self.similarity_thresh = similarity_thresh
        self.labels = None
        self.warm_up_events = warm_up_events
        self.context_attributes = None
        self.value_attributes = None
        self.semantic_attributes = None
        self.distance_thresh = 0.7
        self.min_occurrence_for_df_check = 3
        self.consider_overhead_tasks_separately = False

    def __repr__(self) -> str:
        return "max_buffer_size="+str(self.max_buffer_size) + "-" + "warm_up_events="+str(self.warm_up_events) + "-" + "similarity_thresh="+str(self.similarity_thresh) + "-" + "labels="+str(self.labels)
