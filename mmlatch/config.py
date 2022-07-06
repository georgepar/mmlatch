import argparse
import os
from collections import defaultdict
from pathlib import Path

from mmlatch.util import yaml_load

BASE_PATH = Path(__file__).parent.parent.absolute()

def _nest(d):
    nested = defaultdict(dict)
    for key, val in d.items():
        if "." in key:
            splitkeys = key.split(".")
            inner = _nest({".".join(splitkeys[1:]): val})
            if inner is not None:
                nested[splitkeys[0]].update(inner)
        else:
            if val is not None:
                nested[key] = val
    return dict(nested) if nested else None


def augment_parser(parser):
    """Augment a parser with a set of sane default args
    Args are None by default to allow for setting from YAML config files
    Scripts are primarily configured using YAML config files
    If a CLI argument is passed it takes precedence
    For example:
        my_experiment.yaml
        ------------------
        learning_rate: 1e-3
        batch_size: 32
        python my_experiment.py
    """
    parser.add_argument(
        "--train", default=False, action="store_true", help="Run training"
    )

    parser.add_argument(
        "--test", default=False, action="store_true", help="Run evaluation"
    )

    parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--overfit-batch",
        default=False,
        action="store_true",
        help="Debug: Overfit a single batch to verify model can learn and "
        " gradients propagate. In the end loss should be close to 0",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug: Run a few epochs of training and validation on a couple "
        "of batches to verify everything is in place",
    )

    # parser.add_argument(
    #     '--run-local', dest='run_slurm',
    #     default=None, action='store_false',
    #     help='Run on local machine or server. Turn tqdm and tensorboard on')

    # parser.add_argument(  # Not supported yet
    #     '--run-slurm', dest='run_slurm',
    #     default=None, action='store_true',
    #     help='Run on cluster. Turn off tqdm and tensorboard. '
    #          'Enable slurm specific ops')

    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        type=str,
        default=None,
        help="Device to run on. [cuda|cpu]",
    )

    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Path to data directory",
    )

    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        type=str,
        default=None,
        help="Path to cache directory",
    )

    parser.add_argument(
        "--logging-dir",
        dest="logging_dir",
        type=str,
        default=None,
        help="Path to experiment logging directory",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        dest="optimizer.learning_rate",
        type=float,
        default=None,
        help="Set the learning rate",
    )

    parser.add_argument(
        "-emb",
        "--embeddings-path",
        dest="embeddings.path",
        type=str,
        default=None,
        help="Path to word embeddings",
    )

    parser.add_argument(
        "--embeddings-dim",
        dest="embeddings.dim",
        type=int,
        default=None,
        help="Dimensionality of word embeddings",
    )

    parser.add_argument(
        "--embeddings-dropout",
        dest="embeddings.dropout",
        type=float,
        default=None,
        help="Embeddings Dropout",
    )

    parser.add_argument(
        "--embeddings-finetune",
        dest="embeddings.finetune",
        action="store_true",
        help="Finetune embeddings",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        dest="dataloaders.batch_size",
        type=int,
        default=None,
        help="Set the batch size",
    )

    parser.add_argument(
        "-j",
        "--num-workers",
        dest="dataloaders.num_workers",
        type=int,
        default=None,
        help="Number of workers for data loading",
    )

    parser.add_argument(
        "--pin-memory",
        dest="dataloaders.pin_memory",
        default=None,
        action="store_true",
        help="Pin CUDA memory for data loading",
    )

    parser.add_argument(
        "--accumulation-steps",
        dest="trainer.accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps to simulate large batch size",
    )

    parser.add_argument(
        "--clip-grad-norm",
        dest="trainer.clip_grad_norm",
        type=float,
        default=None,
        help="Gradient accumulation steps to simulate large batch size",
    )

    parser.add_argument(
        "--patience",
        dest="trainer.patience",
        type=int,
        default=None,
        help="Patience for early stopping",
    )

    # Unused for now. When we support early stopping for other metrics
    # We will enable them
    # parser.add_argument(
    #     '--earlystop-metric',  # dest='trainer.earlystop_metric
    #     type=str, default=None,
    #     help='Metric for early stopping')

    # parser.add_argument(
    #     '--earlystop-mode',
    #     type=str, default=None,
    #     help='Mode for early stopping. [min|max]')

    parser.add_argument(
        "-e",
        "--max-epochs",
        dest="trainer.max_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs to run",
    )

    parser.add_argument(
        "-v",
        "--validate-every",
        dest="trainer.validate_every",
        type=int,
        default=None,
        help="Validate every N epochs",
    )

    parser.add_argument(
        "--parallel",
        dest="trainer.parallel",
        default=None,
        action="store_true",
        help="Run on all available GPUs",
    )

    parser.add_argument(
        "--non-blocking",
        dest="trainer.non_blocking",
        default=None,
        action="store_true",
        help="Load data into GPU without blocking",
    )

    parser.add_argument(
        "--retain-graph",
        dest="trainer.retain_graph",
        default=None,
        action="store_true",
        help="Retain computational graph. Might want to set for RNNs",
    )

    parser.add_argument(
        "-exp",
        "--experiment",
        dest="trainer.experiment_name",
        type=str,
        default=None,
        help="Experiment name",
    )

    parser.add_argument(
        "--checkpoint-dir",
        dest="trainer.checkpoint_dir",
        type=str,
        default=None,
        help="Path to checkpoint directory",
    )

    parser.add_argument(
        "--load-model",
        dest="trainer.model_checkpoint",
        type=str,
        default=None,
        help="Resume from model checkpoint / Transfer learning",
    )

    parser.add_argument(
        "--load-optimizer",
        dest="trainer.optimizer_checkpoint",
        type=str,
        default=None,
        help="Resume optimizer",
    )

    return parser


def get_cli(parser):
    parser = augment_parser(parser)
    arguments = parser.parse_args()
    return _nest(vars(arguments))


def default_cli():
    parser = argparse.ArgumentParser(description="CLI parser for experiment")
    return get_cli(parser)


SANE_DEFAULTS = {
    "device": "cpu",
    "train": True,
    "test": True,
    "data_dir": os.path.join(BASE_PATH, "data"),
    "cache_dir": os.path.join(BASE_PATH, "cache"),
    "logging_dir": os.path.join(BASE_PATH, "logs"),
    "optimizer": {
        "name": "Adam",
        "learning_rate": 1e-3,
    },
    "dataloaders": {
        "num_workers": 1,
        "pin_memory": True,
    },
    "trainer": {
        "accumulation_steps": 1,
        "patience": 5,
        "max_epochs": 100,
        "validate_every": 1,
        "parallel": False,
        "non_blocking": True,
        "retain_graph": False,
        "checkpoint_dir": os.path.join(BASE_PATH, "checkpoints"),
        "model_checkpoint": None,
        "optimizer_checkpoint": None,
    },
}


def _merge(*dicts):
    if len(dicts) == 1:
        return dicts[0]
    merged = dicts[0]
    for d in reversed(dicts[1:]):
        for k, v in d.items():
            merged[k] = v
            if isinstance(v, dict):
                d[k] = _merge(*[subd[k] for subd in dicts if k in subd])
            else:
                continue
    return merged


def load_config(parser=None):
    """Load yaml configuration and overwrite with CLI args if provided
    Configuration file format:
        experiment:
            name: "imdb-glove-rnn-256-bi'
            description: Long experiment description
        embeddings:
            path: "../data/glove.840B.300d.txt"
            dim: 300
        dataloaders:
            batch_size: 32
            pin_memory: True
        models:
            rnn:
                hidden_size: 256
                layers: 1
                bidirectional: True
                attention: True
            classifier:
                in_features: 512
                num_classes: 3
        optimizer:
            name: Adam
            learning_rate: 1e-3
        loss:
            name: CrossEntropyLoss
        trainer:
            patience: 5
            retain_graph: True
    """
    cli_args = default_cli() if parser is None else get_cli(parser)
    config_file = cli_args["config"]  # type: ignore
    cfg = yaml_load(config_file)  # type: ignore
    return _merge(cli_args, cfg, SANE_DEFAULTS)


if __name__ == "__main__":
    cfg = load_config("../../tests/test.yaml")
    import pprint

    pprint.pprint(cfg)
