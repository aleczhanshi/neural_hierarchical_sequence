# neural_hierarchical_sequence

**This version of code has the basic skeleton of the hierarchical structure (without the multi-labeling scheme), but its hyperparameters are not well tuned. Often times those hyperparameters are dependent on the workloads (e.g. length of trace, memory footprint, access patterns etc.), so it is recommended to tune the parameters with your workloads.**

**Steps**

1. Generate traces that contain (PC, address/delta), maybe try LLC access or miss stream. Note that it's possible that if a trace is too long, it may have too many unique addresses that might explode the GPU memory. An access can be represented by its absolute address or the delta from previous address.

2. Go to run.sh
- benchmark: specify the csv file that contains pc, address/delta
- pc_localization: start with 0, predict global stream first. Change it to 1 will genereate labels based on PC.
- step_size: sequence length
- multiple: the ratio between page and offset embeddings
- keep_ratio: dropout ratio
- epoch: number of training epochs.

3. run with "sh run.sh"
