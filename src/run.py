import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from gner_trainer import GNERTrainer
from gner_collator import DataCollatorForGNER
from gner_evaluator import compute_metrics


# off wandb
os.environ['WANDB_DISABLED'] = "True"
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the train/dev/test splits and labels."}
    )
    no_load_gner_customized_datasets: bool = field(
        default=False, metadata={"help": "Whether to load GNER datasets. If False, you should provide json files"}
    )
    train_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the train data."}
    )
    valid_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the valid data."}
    )
    test_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the test data."}
    )
    data_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=648,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=648,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    if not data_args.no_load_gner_customized_datasets:
        raw_datasets = load_dataset(
            os.path.join(CURRENT_DIR, "gner_dataset.py"),
            data_dir=data_args.data_dir,
            instruction_file=data_args.instruction_file,
            data_config_dir=data_args.data_config_dir,
            add_dataset_name=data_args.add_dataset_name,
            trust_remote_code=True,
        )
        raw_datasets.cleanup_cache_files()
    else:
        raw_datasets = {}

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    is_encoder_decoder = config.is_encoder_decoder
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    MODEL_CLASS = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
    model = MODEL_CLASS.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    def preprocess_function(example):
        # remove pairs where at least one record is None
        inference = example['split'] != "train"
        if is_encoder_decoder:
            model_inputs = tokenizer(
                text=example['instance']['instruction_inputs'],
                max_length=data_args.max_source_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )
            if not inference:
                model_inputs["labels"] = tokenizer(
                    text_target=example['instance']['prompt_labels'],
                    max_length=data_args.max_target_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )['input_ids']
        else:
            prompt = f"[INST] {example['instance']['instruction_inputs']} [/INST]"
            full_instruction = f"{prompt} {example['instance']['prompt_labels']}"
            max_length = data_args.max_source_length + data_args.max_target_length
            if inference:
                model_inputs = tokenizer(
                    text=prompt,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )
                # Remove the last token if it is an eos token
                if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
                    model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
                    model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]
            else:
                model_inputs = tokenizer(
                    text=full_instruction,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )

                if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
                    model_inputs["input_ids"].append(tokenizer.eos_token_id)
                    model_inputs["attention_mask"].append(1)

                model_inputs["labels"] = model_inputs["input_ids"].copy()

                # Find the prompt length
                prompt = tokenizer(
                    text=prompt,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )["input_ids"]

                # Remove the last token if it is an eos token
                if prompt[-1] == tokenizer.eos_token_id:
                    prompt = prompt[:-1]

                if len(prompt) > len(model_inputs["labels"]):
                    raise ValueError(
                        f"Prompt is longer than the input, something went wrong. Prompt: {prompt}, input:"
                        f" {model_inputs['input_ids']}"
                    )

                for i in range(len(prompt)):
                    model_inputs["labels"][i] = -100

        return model_inputs

    if training_args.do_train:
        if data_args.train_json_dir is not None:
            train_dataset = load_dataset("json", data_files=data_args.train_json_dir, split="train")
            logger.info(f"Use {data_args.train_json_dir} as train dataset, len(dataset) = {len(train_dataset)}")
        elif "train" in raw_datasets:
            train_dataset = raw_datasets["train"]
        else:
            raise ValueError("--do_train requires a train dataset")

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"len(dataset) = {len(train_dataset)}")

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if data_args.valid_json_dir is not None:
            eval_dataset = load_dataset("json", data_files=data_args.valid_json_dir, split="train")
            logger.info(f"Use {data_args.valid_json_dir} as valid dataset, len(dataset) = {len(eval_dataset)}")
        elif "valid" in raw_datasets:
            eval_dataset = raw_datasets["validation"]
        else:
            raise ValueError("--do_eval requires a validation dataset")

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.test_json_dir is not None:
            predict_dataset = load_dataset("json", data_files=data_args.test_json_dir, split="train")
            logger.info(f"Use {data_args.test_json_dir} as predict dataset, len(dataset) = {len(predict_dataset)}")
        elif "test" in raw_datasets:
            predict_dataset = raw_datasets["test"]
        else:
            raise ValueError("--do_predict requires a test dataset")

        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Construct Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForGNER(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        label_pad_token_id=label_pad_token_id,
        return_tensors="pt",
    )

    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    def compute_ner_metrics(dataset, preds, save_prefix=None):
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if not is_encoder_decoder:
            match_pattern = "[/INST]"
            for i, preds in enumerate(decoded_preds):
                decoded_preds[i] = preds[preds.find(match_pattern) + len(match_pattern):].strip()

        all_examples = [example.copy() for example in dataset]
        for idx, decoded_pred in enumerate(decoded_preds):
            all_examples[idx]["prediction"] = decoded_pred

        results = compute_metrics(all_examples, tokenizer=tokenizer)
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_text_generations.jsonl"), "w") as fout:
                for example in all_examples:
                    fout.write(json.dumps(example) + "\n")
        return results

    # Initialize our Trainer
    trainer = GNERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics if training_args.predict_with_generate else None,
    )

    all_metrics = {"run_name": training_args.run_name}
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)


if __name__ == "__main__":
    main()