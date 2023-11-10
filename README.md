# Speech Separation

## Installation guide

1. `pip install -r requirements.txt`
2. Download Librispeech and create Mixture dataset
```shell
python create_dataset.py -c create_dataset.json
```
3. Download my speech separation checkpoint `link`
4. Optional, if you want measure WER and CER, download my audio speech recognition checkpoint `link`.

## Train 

1. `python train.py -c path_to_config`, for example, `python train.py -c src/configs/kaggle.json`.

## Test

1. 


## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_src_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_src_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
