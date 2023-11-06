import argparse
import json
import logging
from typing import Dict

import src.datasets
from src.base.base_text_encoder import BaseTextEncoder
from src.utils.mixture_generator import MixtureGenerator
from src.text_encoder import CTCCharTextEncoder


logger = logging.getLogger(__name__)


def main(config: Dict, text_encoder: BaseTextEncoder):
    for split, params in config["data"].items():
        # set train augmentations
        # if split == 'train':
        #     wave_augs, spec_augs = src.augmentations.from_config(config)
        #     drop_last = True
        # else:
        #     wave_augs, spec_augs = None, None
        #     drop_last = False

        # create and join datasets
        index = []
        for ds in params["datasets"]:
            kwargs = dict(ds["args"])
            kwargs.update({"text_encoder": text_encoder})
            kwargs.update({"config_parser": config})
            dataset = getattr(src.datasets, ds["type"])(**kwargs)
            index += dataset._index
            # wave_augs=wave_augs, spec_augs=spec_augs)
        assert len(index) > 0

        logging.info(split + ": " + str(len(index)))
        mixture_generator = MixtureGenerator(index, **params["mixture_generator_init"])
        mixture_generator.generate_mixes(**params["mixture_generator_generate_mixes"])
        logging.info("finished " + split)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args = args.parse_args()
    with open(args.config, "r") as rfile:
        config = json.load(rfile)

    text_encoder = CTCCharTextEncoder()
    main(config, text_encoder)
