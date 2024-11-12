# LLMs-Powered Real-Time Fault Injection: An Approach to Fault Test Case Generation

This repository is the code implementation of mentioned paper.


# Setup
1. Clone the repository
```
git clone https://github.com/ahmadhatahet/auto-fault-tc-gen.git
cd auto-fault-tc-gen
```

2. Create & activate python environment
```
conda create -n faultgen python==3.10 -y
conda activate faultgen
```

3. Install requirements from source directory
```
pip install -r requirements.in
```

# Directory Structure

__data__:<br>
The dataset and example as excel files.

__imgs__:<br>
Images for the paper.

__notebooks__:<br>
- experiment_bulk: predict target sensor for multiple requirements in each inference.
- experiment_single_drop_sensor: predict target sensor for one requirements in each inference with removed sensor from sensor list.
- experiment_single: predict target sensor for one requirements in each inference.
- results_analysis_instance: analyzing inaccurate prediction for a sensor from multiple models.
- results_analysis_single: analyzing the results and generating tables and plots for the paper.

__results__:<br>
The messages regarding each run using different parameter (refer to paper for more details).

__testgen__:<br>
The python package used for this experiment.
Install it using the command:
```
pip install -e .
```


# Paper Abstract

The most widely recognized testing method for the real-time validation of automotive software systems (ASSs) is Fault Injection (FI).

In accordance with the ISO 26262 standard, the faults are introduced artificially for the purpose of analyzing the safety properties and mechanisms during the development phase.

However, the current FI method and tools have a significant limitation in that they require manual identification of fault space attributes, including fault type, location and time. This is a costly, time-consuming and effort-intensive process.

To address the aforementioned challenge, the present paper proposes a novel LLMs-assisted approach towards the generation of fault test cases (TCs) for utilization during real-time FI tests.  

In order to achieve this, the ability of various LLMs to create TCs that meet the criteria of representativeness and coverage has been investigated. To validate the proposed approach, a high-fidelity system model from the automotive domain was employed as a case study.

The superiority of the proposed approach utilizing GPT-4O-mini in comparison to other state-of-the-art models has been demonstrated with an F1-score of 93\%. This novel approach offers a means of optimizing the testing process, thereby reducing costs while simultaneously enhancing the safety properties of complex safety-critical ASSs.