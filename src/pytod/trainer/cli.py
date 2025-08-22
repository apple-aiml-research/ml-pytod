#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from transformers import Seq2SeqTrainingArguments

from pytod.trainer.utils import infer_data_version_from_path

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/t5-v1_1-base",
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where to store the pretrained models downloaded from huggingface.co"
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use one of the fast tokenizer (backed by the tokenizers"
                " library) or not."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag name or"
                " commit id)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if"
                " `max_source_length` exceeds the model's position embeddings."
            )
        },
    )

    def __post_init__(self):
        assert (
            self.cache_dir is not None
        ), "Please specify a directory where huggingface will save the model artefacts"
        os.environ["WANDB_CACHE_DIR"] = f"{self.cache_dir}/wandb"


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    discard_truncated_examples: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to discard examples with inputs exceeding the max_length when"
                " tokenized."
            )
        },
    )
    source_column: Optional[str] = field(
        default="prompt",
        metadata={
            "help": (
                "The name of the column in the datasets containing the dialogue"
                " context."
            )
        },
    )
    target_column: Optional[str] = field(
        default="completion",
        metadata={
            "help": (
                "The name of the column in the datasets containing the belief state."
            )
        },
    )
    train_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge)"
                " on (a jsonlines or csv file)."
            )
        },
    )
    validation_template_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If --predict_with_generate flag is activated, the dev joint goal"
                " accuracy is computed as the model trains. This is the full path to"
                " the directory containing SGD-formatted dialogues for validation set"
                " without annotations, that are populated with predictions during model"
                " evaluation steps. This should be the same SGD split as the one from"
                " which the --validation_file was derived"
            )
        },
    )
    validation_ref_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If --predict_with_generate flag is activated, the dev joint goal"
                " accuracy is computed as the model trains. This is the full path to"
                " the directory containing the SGD references for the same split from"
                " which --validation_file was derived."
            ),
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input test data file to evaluate the joint goal accuracy"
                " on dev set during training andfor testing the model in inference"
                " mode."
            )
        },
    )
    test_template_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If --do_predict flag is activated, the dev joint goal accuracy is"
                " computed at the end of training. This is the full path to the"
                " directory containing SGD-formatted dialogues without annotations,"
                " that are populated with predictions during inference. This should be"
                " the same SGD split as the one from which the --test_file was derived"
            )
        },
    )
    test_ref_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If --do_predict flag is activated, the dev joint goal accuracy is"
                " computed at the end of training. This is the full path to the"
                " directory containing the SGD references, used to compute the metrics."
            ),
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences"
                " longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization."
                " Sequences longer than this will be truncated, sequences shorter will"
                " be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after"
                " tokenization. Sequences longer than this will be truncated, sequences"
                " shorter will be padded. Will default to `max_target_length`. This"
                " argument is also used as the default value of the"
                " ``generation_max_length`` param of Trainer."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. If False,"
                " will pad the samples dynamically when batching to the maximum length"
                " in the batch. More efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of"
                " training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of"
                " evaluation examples to this value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of"
                " prediction examples to this value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to ignore the tokens corresponding to padded labels in the"
                " loss computation or not."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    override_eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Set to 'validation' to run decoding offline on dev set."},
    )

    def __post_init__(self):
        if self.test_file is None:
            assert isinstance(self.train_files, list)
        if self.train_files is None and (
            self.validation_file is None and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_files is not None:
                for f in self.train_files:
                    assert Path(f).name in {
                        "train.parquet",
                        "train_nlu.parquet",
                        "train_nlu_first.parquet",
                    }, f"Wrong path for train file {self.train_files}"
            if self.validation_file is not None:
                assert (
                    Path(self.validation_file).name == "dev.parquet"
                ), f"Wrong path for validation file {self.validation_file}"
            if self.test_file is not None:
                if self.override_eval_split:
                    assert Path(self.test_file).name == "dev.parquet", (
                        f"Wrong path for test file {self.test_file}. Expected "
                        f"'dev.parquet' since override_eval_split={self.override_eval_split}"
                    )
                else:
                    assert (
                        Path(self.test_file).name == "test.parquet"
                    ), f"Wrong path for test file {self.test_file}"
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(
        default="models",
        metadata={
            "help": (
                "The output directory where the model predictions and checkpoints will"
                " be written."
            )
        },
    )
    hyps_dir: str = field(
        default="hyps",
        metadata={
            "help": (
                "Directory where the SGD-formatted hypotheses and frame-level evaluator"
                " outputs are to be saved."
            )
        },
    )
    metrics_dir: str = field(
        default="metrics",
        metadata={"help": "Directory where the SGD evaluator outputs are to be saved."},
    )
    experiment_name: str = field(
        default=None,
        metadata={
            "help": (
                "The name of the experiment to be trained. This will be used to create"
                " a hierarchy for checkpointing, saving SGD-formatted predictions and"
                " SGD evluator ouputs."
            ),
        },
    )
    experiment_name_suffix: str = field(
        default="",
        metadata={
            "help": (
                "The suffix to be added to the experiment name. This will be used when"
                "creating hypotheses and metrics directories, to support decoding the"
                "same models with different settings."
            ),
        },
    )
    eval_steps: int = field(
        default=5000,
        metadata={
            "help": (
                "Number of update steps between two evaluations if"
                " `evaluation_strategy='steps'`. Will default to the same value as"
                " `logging_steps` if not set."
            )
        },
    )
    save_strategy: str = field(
        default="steps",
        metadata={
            "help": (
                "The save strategy to adopt during training. Possible values are 'no'"
                " (no save is done during training)/'steps'(save is done and logged"
                " every `save_steps`)/'epoch' (evaluation is done at the end of each"
                " epoch."
            )
        },
    )
    save_steps: int = field(
        default=5000,
        metadata={
            "help": (
                "Number of updates steps before two checkpoint saves if"
                " `save_strategy='steps'`."
            )
        },
    )
    metric_for_best_model: str = field(
        default="loss",
        metadata={
            "help": (
                "Use in conjunction with `load_best_model_at_end` to specify the metric"
                " to use to compare two differentmodels. Must be the name of a metric"
                " returned by the evaluation with or without the prefix `'eval_'`."
                " Will default to `'loss'` if unspecified and"
                " `load_best_model_at_end=True` (to use the evaluation loss).If you set"
                " this value, `greater_is_better` will default to `True`. Don't forget"
                " to set it to `False` if your metric is better when lower."
            )
        },
    )
    greater_is_better: str = field(
        default=False,
        metadata={"help": ""},
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={
            "help": (
                "Used to stop training when the metric specified in"
                " `metric_for_best_model` fails to improve forthe specified number of"
                " evaluation calls."
            ),
        },
    )
    report_to: str = field(default=None, metadata={"help": ""})
    seed: int = field(
        default=230792,
        metadata={
            "help": (
                " Random seed that will be set at the beginning of training. To ensure"
                " reproducibility across runs, use the model_init function to"
                " instantiate the model if it has some randomly initialized parameters."
                " Note that the same seed is applied to the random generators within"
                " the data sampling."
            )
        },
    )
    ensure_determinism: bool = field(
        default=True,
        metadata={
            "help": (
                "If this variable is set to True, the all the steps described in"
                " https://pytorch.org/docs/stable/notes/randomness.html are followed."
                " Note that we defer set_seed to `huggingface` and only ensure that"
                " pytorch uses deterministic implementations, disables convolution"
                " benchmarking and run cuda in debug mode to allow deterministic"
                " behaviour."
            ),
        },
    )
    lr_log_freq: int = field(
        default=100, metadata={"help": "Learning rate logging frequency in steps."}
    )
    log_lr_freq_limit: int = field(
        default=5000,
        metadata={
            "help": (
                "Log learning rate with `lr_log_freq` until this many steps have been"
                " completed. Logged every `logging_steps` gradient steps thereafter"
            )
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "This the ``wandb`` entity where the runs are logged to. Check the"
                " project workspace in your dashboard to see what this is."
            )
        },
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "`wandb` project where the logs are made."}
    )
    wandb_log_model: Optional[str] = field(
        default="true",
        metadata={
            "help": (
                "If `true`, saves the best model if `log_best_model=True` in `wandb`"
                " storage."
            )
        },
    )
    wandb_user_name: Optional[str] = field(
        default=None,
        metadata={"help": "Must be set to ``wandb`` user name if `report_to=wandb`."},
    )
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Set tags for a ``wandb`` run to simplify data analysis.",
        },
    )
    data_variant: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "SGD schema variant. This is automatically inferred from"
            "the `validation_file` or `test_file` data arguments.",
        },
    )

    def __post_init__(self):
        logging.info("Running sanity checks on arguments")
        super().__post_init__()
        self._validate_and_modify_arguments()
        if self.ensure_determinism:
            logging.info(
                "Setting debug environment for CUDA to ensure determinism and avoiding"
                " non-deterministic ops"
            )
            logging.info("Seed will be set by the trainer")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            logging.info(
                f"CUDA_WORK_SPACE_CONFIG: {os.environ.get('CUDA_WORK_SPACE_CONFIG')}"
            )
            torch.use_deterministic_algorithms(True)
            # Enable CUDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _validate_and_modify_arguments(self):
        validate_and_modify_train_arguments(self)


def validate_and_modify_train_arguments(training_args: CustomSeq2SeqTrainingArguments):
    """Validates the input arguments and modifies them in certain scenarios to simplify
    experiment setup.
    """
    # experiment_name defaults to "" - ensure it is set
    if training_args.experiment_name == "":
        if training_args.resume_from_checkpoint:
            training_args.experiment_name = None
        else:
            raise ValueError(
                "Experiment name cannot be '', please set experiment_name in the training"
                " arguments/"
            )
    model_input_data_version = None
    if training_args.experiment_name is None:
        if not training_args.resume_from_checkpoint:
            if not training_args.do_predict:
                raise ValueError(
                    "You must provide a descriptive experiment name to create output"
                    " directory hierarchy"
                )
        if training_args.resume_from_checkpoint or training_args.do_predict:
            model_input_data_version = infer_data_version_from_path(
                str(training_args.output_dir)
            )
            if model_input_data_version in Path(training_args.output_dir).name:
                training_args.experiment_name = Path(
                    training_args.output_dir
                ).parent.name
            else:
                assert (
                    model_input_data_version
                    in Path(training_args.output_dir).parent.name
                ), (
                    "Expected to find version in parent of"
                    f" {Path(training_args.output_dir)}"
                )
                training_args.experiment_name = Path(
                    training_args.output_dir
                ).parent.parent.name
            msg = (
                "Restarting from checkpoint"
                if training_args.resume_from_checkpoint
                else "Doing inference"
            )
            logging.info(
                f"{msg}, inferred experiment name: {training_args.experiment_name}"
            )
            logging.info(
                f"{msg}, inferred model input data version: {model_input_data_version}"
            )
    else:
        if training_args.resume_from_checkpoint:
            training_args.experiment_name = (
                f"seed_{training_args.seed}_{training_args.experiment_name}"
            )
            logging.info("Restarting from checkpoint...")
    # prefix experiment name with the seed when training
    if not (training_args.resume_from_checkpoint or training_args.do_predict):
        training_args.experiment_name = (
            f"seed_{training_args.seed}_{training_args.experiment_name}"
        )
        logging.info(
            "Prefixed experiment name with random seed:"
            f" {training_args.experiment_name}"
        )
    # set the wandb run name to be the same as the experiment name
    # this is suffixed with _version_* at training time to differentiate
    # between same experiment ran on different input data versions
    training_args.run_name = training_args.experiment_name
    if model_input_data_version is not None and model_input_data_version:
        training_args.run_name = f"{training_args.run_name}_{model_input_data_version}"
    logging.info(f"Logs will appear in wandb under run name: {training_args.run_name}")
    # there will be errors in SGD evaluation if do_predict/do_eval are both active
    # and this is never necessary since evaluation runs during model training
    if training_args.do_predict:
        if training_args.do_eval:
            logging.warning(
                "Inference mode is designed to be standalone but got do_eval=True due"
                " to default evaluation strategy setting. Disabling evaluation on"
                " validation data."
            )
            training_args.do_eval = False
    # no need to log more frequently then we evaluate
    if training_args.logging_steps != training_args.eval_steps:
        logging.warning(
            f"Logging frequency is {training_args.logging_steps} but evaluation"
            f" frequency is {training_args.eval_steps}. Setting logging frequency to"
            f" {training_args.eval_steps} steps."
        )
        training_args.logging_steps = training_args.eval_steps
    # setup wandb if used
    if (
        isinstance(training_args.report_to, list) and "wandb" in training_args.report_to
    ) or training_args.report_to == "all":
        logging.info(
            "Make sure you are logged in to ``wandb`` using wandb login and the API key"
            " for your project!"
        )
        try:
            import wandb

            assert (
                training_args.wandb_entity is not None
            ), "Please check your wandb project and provide the entity name"
            assert (
                training_args.wandb_project is not None
            ), "Please check your wandb project and provide the project name"
            # offer the option to delete existing runs if we are re-running an existing
            # experiment
            if training_args.overwrite_output_dir:
                wandb_api = wandb.Api()
                to_delete_runs = wandb_api.runs(
                    path=f"{training_args.wandb_entity}/{training_args.wandb_project}",
                    filters={"config.run_name": f"{training_args.run_name}"},
                )
                if to_delete_runs:
                    user_confirmation = input(
                        f"::WARNING:: You are overriding {training_args.output_dir}. "
                        "Would you also like to delete the ``wandb`` runs? (y/n)"
                    )
                    if user_confirmation == "y":
                        for r in to_delete_runs:
                            r.delete()
                    else:
                        if user_confirmation != "n":
                            logging.info(f"Unknown option: {user_confirmation}")
                        else:
                            logging.info(
                                f"Run {training_args.run_name} will not be deleted from"
                                " ``wandb``."
                            )
        except ModuleNotFoundError:
            pass
