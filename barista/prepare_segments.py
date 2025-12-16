"""Script to preprocess and prepare data segments.

Example usage:
    python prepare_segments.py --config config/braintreebank_config.yaml --experiment sentence_onset
"""

import argparse
from omegaconf import OmegaConf

from barista.data.braintreebank_wrapper import BrainTreebankWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, type=str, help="path to config for segmentation")
    parser.add_argument("--experiment", required=True, type=str, help="experiment to segment data for")
    
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    config = OmegaConf.load(args.config)

    ## Instantiating BrainTreebankWrapper will be default handle all preprocessing.
    ## If preprocessing is complete, then the dataset will be ready to use for training.
    config.experiment = args.experiment
    print(f"Segmenting data for experiment {args.experiment}")
    braintreebank_wrapper = BrainTreebankWrapper(config, only_segment_generation=True)
