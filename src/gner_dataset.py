import json
import os
import random
import datasets
import pandas as pd
import csv


logger = datasets.logging.get_logger(__name__)

# modified from https://github.com/BeyonderXX/InstructUIE/blob/master/src/uie_dataset.py
class GNERConfig(datasets.BuilderConfig):
    """
    GNERDataset condig
    """

    def __init__(
        self,
        *args,
        data_dir=None,
        instruction_file=None,
        data_config_dir=None,
        add_dataset_name=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.instructions = self._parse_instruction(instruction_file)
        self.data_configs = self._parse_data_config(data_config_dir)
        self.add_dataset_name = add_dataset_name

    def _parse_instruction(self, instruction_file):
        """
        Instruction file example:
        [
            "Please analyze the sentence provided, list the inherent entities\n",
            "List all the entities in the sentence"
        ]
        """
        if not instruction_file:
            return None
        instructions = []

        with open(instruction_file, 'r+') as f:
            instructions = json.load(f)
        return instructions

    def _parse_data_config(self, data_config_dir):
        """
        Task config file example:
        [
            {
                "dataset name": "mit-movie",
                "sampling strategy": "random",
                "max_num_instances_per_task": 200,
                "over_sampling": false
            },
            {
                "dataset name": "mit-restaurant",
                "sampling strategy": "random",
                "max_num_instances_per_task": 200,
                "over_sampling": false
            }
        ]
        """
        if not data_config_dir:
            return None

        data_configs = {}
        for split in ["train", "dev", "test"]:
            file_name = f"{split}_configs.json"
            data_config_file = os.path.join(data_config_dir, file_name)

            if not os.path.exists(data_config_file):
                raise ValueError('Please check {} split, {} not exists!'.format(split, data_config_file))

            with open(data_config_file, 'r+') as f:
                data_configs[split] = json.loads(f.read())

        return data_configs


class GNERDataset(datasets.GeneratorBasedBuilder):
    """GNER Datasets"""

    BUILDER_CONFIG_CLASS = GNERConfig


    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "dataset": datasets.Value("string"),
                    "split": datasets.Value("string"),
                    "label_list": datasets.Sequence(datasets.Value("string")),
                    "instance": {
                        "id": datasets.Value("string"),
                        "words": datasets.Sequence(datasets.Value("string")),
                        "labels": datasets.Sequence(datasets.Value("string")),
                        "instruction_inputs": datasets.Value("string"),
                        "prompt_labels": datasets.Value("string"),
                    }
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.data_configs is None:
            logger.error("Please provide right input: data_dir or data_config_dir!")

        data_configs = self.config.data_configs

        split_generators = []
        if len(data_configs['train']) > 0:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_config": data_configs['train'],
                        "add_dataset_name": self.config.add_dataset_name,
                        "split": "train"
                    }
                )
            )
        if len(data_configs['dev']) > 0:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_config": data_configs['dev'],
                        "add_dataset_name": self.config.add_dataset_name,
                        "split": "dev"
                    }
                )
            )
        if len(data_configs['test']) > 0:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_config": data_configs['test'],
                        "add_dataset_name": self.config.add_dataset_name,
                        "split": "test"
                    }
                ),
            )

        return split_generators

    # read conll-style dataset
    def _load_dataset(self, dataset_path, labels_path):
        data_df = pd.read_csv(dataset_path, delimiter='\t', quoting=csv.QUOTE_NONE, 
                            skip_blank_lines=False, header=None, keep_default_na=False, na_values=[''], low_memory=False)

        with open(labels_path, "r") as f:
            labels_list = f.read().splitlines()
            assert "O" not in labels_list

        instances = []
        idx, words, labels = 0, [], []
        for row in data_df.values:
            if not pd.isna(row[1]):
                words.append(row[0])
                labels.append(row[-1])
            elif words != []:
                instances.append({"id": idx, "words": words, "labels": labels})
                idx += 1
                words, labels = [], []

        if words != []:
            instances.append({"id": idx, "words": words, "labels": labels})

        return instances, labels_list

    def _get_instruction(self):
        return self.config.instructions[0]

    # generate prompt labels
    def _generate_labeled_string(self, words, labels):
        label_text_list = []
        for word, label in zip(words, labels):
            label_text_list.append(f"{word}({label})")

        return " ".join(label_text_list)

    # sample instances
    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances, over_sampling):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances is not None and over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.extend(origin_instances)
        return instances

    def _generate_examples(self, data_config, add_dataset_name, split):
        """Yields examples."""
        data_dir = self.config.data_dir
        if len(data_config) == 0:
            return
        # Load data from files
        for dataset in data_config:
            # Read info from data_config
            dataset_name = dataset["dataset name"]
            sampling_strategy = dataset.get("sampling strategy", "random")
            max_num_instances = dataset.get("max_num_instances", "full")
            over_sampling = dataset.get("over_sampling", False)
            dataset_path = os.path.join(data_dir, dataset_name, split + ".txt")
            labels_path = os.path.join(data_dir, dataset_name, "label.txt")
            assert os.path.exists(dataset_path)
            assert os.path.exists(labels_path)

            # load data from files
            instances, label_list = self._load_dataset(dataset_path, labels_path)
            instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances, over_sampling)
            for idx, instance in enumerate(instances):
                words, labels = instance["words"], instance["labels"]
                instruction = self._get_instruction()
                random.shuffle(label_list)
                instruction += f"\nUse the specific entity tags: {', '.join(label_list)} and O.\n"
                if add_dataset_name:
                    instruction += f"Dataset: {dataset_name}.\n"
                instruction += "Sentence: " + " ".join(words)
                label_text = self._generate_labeled_string(words, labels)
                yield f"{dataset_name}##{idx}", {
                    "dataset": dataset_name,
                    "split": split,
                    "label_list": label_list,
                    "instance": {
                        "id": str(idx),
                        "words": instance["words"],
                        "labels": instance["labels"],
                        "instruction_inputs": instruction,
                        "prompt_labels": label_text,
                    }
                }
