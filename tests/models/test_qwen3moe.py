from typing_extensions import *
import os

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig
from roundpipe.models.function import (
    CompileForCausalLMLoss,
    ChunkedCompileLinearForCausalLMLoss,
)

MODEL_PATH = (
    os.environ["QWEN3_30B_PATH"]
    if "QWEN3_30B_PATH" in os.environ
    else "Qwen/Qwen3-30B-A3B"
)
DATASET_PATH = (
    os.environ["MATH_500_PATH"]
    if "MATH_500_PATH" in os.environ
    else "HuggingFaceH4/MATH-500"
)
REF_LOSS = [1.886754035949707, 1.853429913520813, 1.995058298110962, 1.4403775930404663, 1.864284634590149, 2.015983819961548, 1.7562131881713867, 1.674486756324768, 1.7251732349395752, 1.4633623361587524]  # fmt: skip
REF_AUX_LOSS = [8.23470687866211, 8.26980209350586, 8.25084114074707, 8.379385948181152, 8.347529411315918, 8.340141296386719, 8.31120777130127, 8.294321060180664, 8.302978515625, 8.271138191223145]  # fmt: skip


def get_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    raw_dataset = load_dataset(DATASET_PATH, split="test")

    samples = []
    for example in raw_dataset:
        example = cast(Dict[str, str], example)
        messages = [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]},
        ]
        conversation = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if len(conversation) < 1000:
            samples.append(conversation)
        if len(samples) >= 100:
            break

    def collate_fn(text):
        input_dict = tokenizer(
            text, return_tensors="pt", padding=True, truncation=False
        )
        labels = input_dict["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        input_dict["labels"] = labels
        return input_dict

    return torch.utils.data.DataLoader(
        samples,  # pyright: ignore[reportArgumentType]
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn,
    )


def test_qwen3moe_auto_forward():
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=4),
        use_sequential_preset=False,
    )
    dataloader = get_dataloader()

    losses = []
    aux_losses = []
    for input_dict in dataloader:
        labels = input_dict.pop("labels")
        out = model(**input_dict)
        aux_losses.append(out.aux_loss)
        loss = CompileForCausalLMLoss(
            logits=out.logits,
            labels=labels,
            vocab_size=model.config.vocab_size,
        )
        losses.append(loss.item())

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS)]
    assert sum(diff) / len(diff) < 0.005


def test_qwen3moe_preset_forward():
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=4),
        use_sequential_preset=True,
    )
    dataloader = get_dataloader()

    losses = []
    aux_losses = []
    for input_dict in dataloader:
        labels = input_dict["labels"]
        num_items_in_batch = (labels[..., 1:] != -100).sum().item()
        out = model(
            **input_dict,
            output_router_logits=True,
            return_logits=False,
            num_items_in_batch=num_items_in_batch
        )
        aux_losses.append(out.aux_loss)
        losses.append(
            ((out.loss - model.router_aux_loss_coef * out.aux_loss) * 4).item()
        )

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS)]
    assert sum(diff) / len(diff) < 0.005
    aux_diff = [abs(v.item() - ref) / ref for v, ref in zip(aux_losses, REF_AUX_LOSS)]
    assert sum(aux_diff) / len(aux_diff) < 0.01
