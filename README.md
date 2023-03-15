# CP_multimodal
## Installation

```
conda create --name multimodal_module python==3.10

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

Get UCP timestamp-scores given a segment (path: scripts/exps)
```
sh get_UCP_scores.sh
```

Prepare data
```
sh prepare_full_data.sh
```

Inference on full data
```
sh prepare_full_data.sh
```

