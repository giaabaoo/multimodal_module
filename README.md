# CP_multimodal
## Installation

```
conda create --name multimodal_module python==3.10
pip install -r requirements.txt
conda install -y ffmpeg
```

## Folder structure
multimodal_module/ </br>
├── README.md </br>
├── configs/ </br>
├── data/ </br> 
├── output/ </br>
├── scripts/ </br>
├── tools/ </br>
├── unsupervised/ </br>

## Usage 
Refer to the scripts folder to run the unsupervised method for changepoint detection.

Get UCP timestamp-scores given a segment (path: scripts/exps) with shape (number of detected tracks, number of timestamps):
```
sh get_UCP_scores.sh
```

### Data preparation (Currently testing on minieval data)
Prepare csv file containing changepoint and non-changepoint segments:
```
sh prepare_minieval_data.sh
```

For faster inference, it is recommended to split the full data into batches (default 1000 segments per batch) and run them in parallel:
```
sh split_batch_training.sh
```

### Inference
Inference on full data using a single annotation file (currently ineffective):
```
sh run.sh
```

Inference on full data using a multiple files (batches). On each machine, specify the batch_idx incrementally. Currently using minieval.yaml for inference on minieval data.
On machine 1: 
```
sh run_batch1.sh
```
On machine 2: 
```
sh run_batch2.sh
```

Inference on positve data only (segments with changepoints):
```
sh run_positive_data.sh
```

### Evaluation
There are two modes of evaluation according to the inference strategies.

Evaluate on minieval batches:
```
sh evaluate_minieval.sh
```

Evaluate on all batches:
```
sh evaluate_all_batches.sh
```

Evaluate on a single file:
```
sh evaluate_single_file.sh
```
