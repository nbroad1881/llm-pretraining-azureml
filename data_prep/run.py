"""
This file will run the tokenizer on all of the json files in the given path.
The tokenized sequences will get grouped into chunks of the specified length, 
and then saved into parquet files in the specified output directory. Each file
will have the specified number of samples in it.

This is meant to be run in Azure ML.
"""


from pathlib import Path
from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from transformers import AutoTokenizer, HfArgumentParser
from datasets import load_dataset


TEXT_COLUMN_NAME = "text"


@dataclass
class DataArguments:
    """
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory where the train json lines files are located."
        },
    )
    eval_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory where the eval json lines files are located."},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": (
                """The maximum total input sequence length after tokenization. 
                All sequences will be concatenated together and then broken into
                chunks this long"""
            )
        },
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the tokenizer to be used."},
    )
    num_proc: int = field(
        default=4,
        metadata={"help": ("Number of processes for the map function.")},
    )
    samples_per_file: int = field(
        default=200_000,
        metadata={
            "help": (
                "Instead of saving one file, save multiple smaller ones with this number of examples in each."
            )
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory where the output files will be stored."},
    )


def main():

    parser = HfArgumentParser(DataArguments)

    data_args = parser.parse_args_into_dataclasses()[0]


    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_path)

    train_files = list(map(str, Path(data_args.train_data_dir).glob("*.json")))
    eval_files = list(map(str, Path(data_args.eval_data_dir).glob("*.json")))

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_files,
            "validation": eval_files,
        },
    )

    def tokenize(examples):
        return tokenizer(examples[TEXT_COLUMN_NAME])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (
            total_length // data_args.max_seq_length
        ) * data_args.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + data_args.max_seq_length]
                for i in range(0, total_length, data_args.max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=data_args.num_proc,
        desc=f"Tokenizing dataset on {data_args.num_proc} processes.",
        remove_columns=dataset["train"].column_names,
    )

    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=data_args.num_proc,
        desc=f"Grouping texts on {data_args.num_proc} processes.",
        remove_columns=dataset["train"].column_names,
    )

    for split in ["train", "validation"]:
        save_dir = Path(data_args.output_dir) / split

        save_dir.mkdir(parents=True, exist_ok=True)
        
        num_total_samples = len(dataset[split])
        for filenum, start_idx in enumerate(
            range(0, num_total_samples, data_args.samples_per_file)
        ):
            end_idx = min(num_total_samples, start_idx + data_args.samples_per_file)

            temp = dataset[split].select(range(start_idx, end_idx))
            temp.to_parquet(save_dir / f"tokenized_{filenum:04}.parquet")


if __name__ == "__main__":
    main()
