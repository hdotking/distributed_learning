import os
from typing import Optional, Tuple

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from peft import LoraConfig
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)


class DatasetCreator:
    """
    Usage:
    creator = DatasetCreator(tokenizer, data_args, training_args)
    train_data, valid_data, test_data = creator.create_datasets()
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args, training_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args

    def process_data(self, examples):
        # Tokenize the input text
        tokenized_inputs = self.tokenizer(
            examples[self.data_args.dataset_text_field],
            truncation=True,
            is_split_into_words=True,
        )

        # Align labels with tokens
        word_ids = tokenized_inputs.word_ids()
        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(examples["ner_tags"][word_idx])

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    def create_datasets(
        self,
    ) -> Tuple[Optional[DatasetDict], Optional[DatasetDict], Optional[DatasetDict]]:
        raw_datasets = {}
        splits = [split.strip() for split in self.data_args.splits.split(",")]
        for split in splits:
            try:
                dataset = load_dataset(self.data_args.dataset_name, split=split)
            except DatasetGenerationError:
                dataset = load_from_disk(
                    os.path.join(self.data_args.dataset_name, split)
                )

            dataset = dataset.map(self.process_data, batched=True)
            raw_datasets[split] = dataset

        train_data = raw_datasets.get("train")
        valid_data = raw_datasets.get("validation")
        test_data = raw_datasets.get("test")

        return train_data, valid_data, test_data


class QuantizationConfigBuilder:
    def __init__(self, args):
        self.args = args
        self.bnb_config = None
        self.quant_storage_dtype = None

    def build(self):
        if self.args.use_4bit_quantization:
            self._build_4bit_config()
        elif self.args.use_8bit_quantization:
            self._build_8bit_config()

        return self.bnb_config

    def _build_4bit_config(self):
        compute_dtype = getattr(torch, self.args.bnb_4bit_compute_dtype)
        self.quant_storage_dtype = getattr(
            torch, self.args.bnb_4bit_quant_storage_dtype
        )

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.args.use_4bit_quantization,
            bnb_4bit_quant_type=self.args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.args.use_nested_quant,
            bnb_4bit_quant_storage=self.quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and self.args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    def _build_8bit_config(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=self.args.use_8bit_quantization
        )


class ModelLoader:
    def __init__(self, args, data_args, quant_config):
        self.args = args
        self.data_args = data_args
        self.quant_config = quant_config

    def load_model(self):
        torch_dtype = (
            self.quant_config.quant_storage_dtype
            if self.quant_config.quant_storage_dtype
            and self.quant_config.quant_storage_dtype.is_floating_point
            else torch.float32
        )
        return AutoModelForTokenClassification.from_pretrained(
            self.args.model_name_or_path,
            quantization_config=self.quant_config.bnb_config,
            trust_remote_code=True,
            attn_implementation=(
                "flash_attention_2" if self.args.use_flash_attn else "eager"
            ),
            torch_dtype=torch_dtype,
        )


class TokenizerLoader:
    def __init__(self, args):
        self.args = args

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


def create_and_prepare_model(args, data_args, training_args):
    # Step 1: Configure Quantisation
    quant_config = QuantizationConfigBuilder(args).build()

    # Step 2: Load Model
    model_loader = ModelLoader(args, data_args, quant_config)
    model = model_loader.load_model()

    # Step 3: Configure Tokeniser
    tokenizer_loader = TokenizerLoader(args)
    tokenizer = tokenizer_loader.load_tokenizer()

    # Step 4: Add PEFT Configuration (Optional)
    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="TOKEN_CLS",
            target_modules=(
                args.lora_target_modules.split(",")
                if args.lora_target_modules != "all-linear"
                else args.lora_target_modules
            ),
        )

    return model, peft_config, tokenizer
