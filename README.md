# Speech Separation

## Installation guide

1. `pip install -r requirements.txt`
2. Download Librispeech and create Mixture dataset
```shell
python create_dataset.py -c create_dataset.json
```
3. Download my speech separation checkpoint `link`
4. Optional, if you want to measure WER and CER, download my audio speech recognition checkpoint `link`.

## Train 

1. `python train.py -c path_to_config`, for example, `python train.py -c src/configs/kaggle.json`.

## Test

0. Make sure that you created dataset and downloaded all needed checkpoints.
1. If your config and checkpoint are placed in test_model/config.json and test_model/checkpoint.pth respectively.
```shell
python test.py
``` 
2. Or you can specify paths and run `python test.py -c path_to_config --ss_checkpoint path_to_ss_checkpoint`
3. If you want to measure speech recognition model quality on my speech separation solution, use test_model/asr_config.json and run `python test.py -c test_model/config_for_asr.json --ss_checkpoint path_to_ss_checkpoint --asr_checkpoint path_to_asr_checkpoint`.
4. If you want to test quality on segmented audio (segmented by 100ms windows on default), use test_model/segmentation_config.json run `python3 test.py -c test_model/segmentation_config.json -s window_len_in_seconds`.


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
