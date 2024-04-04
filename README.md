# Recognizing Task-Level Events from User Interaction Data

<sub>
written by <a href="mailto:rebmann@uni-mannheim.de">Adrian Rebmann</a><br />
</sub>

## About
This repository contains the implementation, data, evaluation scripts, and results as described in the manuscript <i>Recognizing Task-Level Events from User Interaction Data</i> by A. Rebmann and H. van der Aa, submitted to <i>Information Systems</i>.

## Setup and Usage

### Installation instructions
**The project requires python >= 3.9**

1. create a virtual environment for the project 
2. install the project using <code> pip install .</code>
3. install Spacy language model: <code>python -m spacy download en_core_web_md</code>

### Directories
The following default directories are used for input and output.

* Input: <code>data/input</code>
* Output <code>data/output/</code>

### Configuring and running the approach
To run the approach, create a class with the following parameters:

<code>
task_recognizer = TaskRecognizer(max_buffer_size, 
warm_up_events, similarity_thresh,
context_atts, value_atts, semantic_atts, 
consider_overhead_tasks_separately=False, labels=None)
</code>

* <code>max_buffer_size</code> is the maximum number of events that are stored in the buffer
* <code>warm_up_events</code> is the number of events that are used to initialize the approach
* <code>similarity_thresh</code> is the similarity threshold used to assess contextual similarity
* <code>context_atts</code> is a list of context attributes
* <code>value_atts</code> is a list of value attributes
* <code>semantic_atts</code> is a list of semantic attributes
* <code>consider_overhead_tasks_separately</code> is a boolean indicating whether overhead tasks should be considered separately
* <code>labels</code> is a list of labels for the task types (optional)

## Evaluation
### Results from the paper and additional results
The results reported in the paper can be obtained using the Python notebook in <code>notebooks/evaluation.ipynb</code>. It also contains additional results that we could not include in the paper due to space reasons.
### Data
Our approach was tested on a collection of user interaction logs covering real task execution recordings.
The raw data and logs used are located in <code>input/logs</code>, we used the script <code>util/data_util.py</code> to create the logs by randomly merging instances of different task types.
### Reproduce
To obtain all results four our approach, run the evaluation script using <code>python evaluation.py</code>.
To obtain the results for the baselines, go to [this repository](https://gitlab.uni-mannheim.de/processanalytics/task-recognition-from-event-stream), which contains our previous approach implementation and the baseline implementations, and follow the instructions in the README.

* L1 is based on data from  [Leno et al.](https://doi.org/10.6084/m9.figshare.12543587)
* L2 is based on data from [Leno et al.](https://doi.org/10.6084/m9.figshare.12543587) and [Agostinelli et al.](https://gitlab.uni-mannheim.de/processanalytics/task-recognition-from-event-stream/-/blob/main/logs/raw/agostinelli.xes) 
* L3 is based on data from Abb & Rehse described [here](https://link.springer.com/chapter/10.1007/978-3-031-16103-2_7).

