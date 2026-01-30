from typing_extensions import *
import os
import itertools
import pathlib
import pickle

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig, GradScaler
from roundpipe.optim import Adam
from roundpipe.models.function import (
    CompileForCausalLMLoss,
    ChunkedCompileLinearForCausalLMLoss,
)

MODEL_PATH = (
    os.environ["QWEN3_0_6B_PATH"]
    if "QWEN3_0_6B_PATH" in os.environ
    else "Qwen/Qwen3-0.6B"
)
DATASET_PATH = (
    os.environ["MATH_500_PATH"]
    if "MATH_500_PATH" in os.environ
    else "HuggingFaceH4/MATH-500"
)
with open(pathlib.Path(__file__).parent / "test_qwen3_ref.pkl", "rb") as f:
    REF_LOSS: Dict[bool, List[float]] = pickle.load(f)


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
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )


@pytest.mark.parametrize("use_preset", [True, False])
def test_qwen3_miniumn(use_preset: bool):
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float32,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=3),
        use_sequential_preset=use_preset,
    )

    optim = torch.optim.Adam(raw_model.parameters(), lr=1e-5, fused=True)
    dataloader = get_dataloader()

    losses = []
    for epoch in range(2):
        for input_dict in dataloader:
            labels = input_dict.pop("labels")
            logits = model(**input_dict).logits
            loss = CompileForCausalLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=model.config.vocab_size,
            )
            loss.backward()
            losses.append(loss.item())
            model.synchronize()
            optim.step()
            optim.zero_grad()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS[False])]
    assert sum(diff) / len(diff) < 0.005


@pytest.mark.parametrize(
    "use_preset, is_async", itertools.product([True, False], [True, False])
)
def test_qwen3_classic(use_preset: bool, is_async: bool):
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=3),
        use_sequential_preset=use_preset,
        optim_dtype=torch.float32,
    )

    optim = Adam(model.optim_parameters(), lr=1e-5)
    scaler = GradScaler(8192.0)
    dataloader = get_dataloader()

    losses = []
    for epoch in range(2):
        for input_dict in dataloader:
            labels = input_dict.pop("labels")
            logits = model(**input_dict).logits
            loss = CompileForCausalLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=model.config.vocab_size,
            )
            scaler.scale(loss).backward()
            losses.append(loss.item())
            model.step(
                lambda: (scaler.step(optim), optim.zero_grad(), None)[-1],
                is_async=is_async,
            )
            scaler.update()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS[is_async])]
    assert sum(diff) / len(diff) < 0.01


def test_qwen3_classic_accumulate():
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=1),
        use_sequential_preset=False,
        optim_dtype=torch.float32,
    )

    optim = Adam(model.optim_parameters(), lr=1e-5)
    scaler = GradScaler(8192.0)
    dataloader = get_dataloader()

    losses = []
    for epoch in range(2):
        for input_dict in dataloader:
            input_ids = input_dict["input_ids"].split(1)
            attention_mask = input_dict["attention_mask"].split(1)
            labels = input_dict["labels"].split(1)
            num_items_in_batch = (input_dict["labels"][..., 1:] != -100).sum().item()
            loss = 0.0
            for mb_input_ids, mb_attention_mask, mb_labels in zip(
                input_ids, attention_mask, labels
            ):
                mb_input_dict = {
                    "input_ids": mb_input_ids,
                    "attention_mask": mb_attention_mask,
                }
                mb_logits = model(**mb_input_dict).logits
                mb_loss = CompileForCausalLMLoss(
                    logits=mb_logits,
                    labels=mb_labels,
                    vocab_size=model.config.vocab_size,
                    num_items_in_batch=num_items_in_batch,
                )
                scaler.scale(mb_loss).backward()
                loss += mb_loss.item()
            losses.append(loss)
            model.step(
                lambda: (scaler.step(optim), optim.zero_grad(), None)[-1],
            )
            scaler.update()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS[True])]
    assert sum(diff) / len(diff) < 0.01


@pytest.mark.parametrize("is_async", [True, False])
def test_qwen3_fused(is_async: bool):
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=3),
        use_sequential_preset=True,
        optim_dtype=torch.float32,
    )

    optim = Adam(model.optim_parameters(), lr=1e-5)
    scaler = GradScaler(8192.0)
    dataloader = get_dataloader()

    losses = []
    for epoch in range(2):
        for input_dict in dataloader:
            labels = input_dict["labels"]
            num_items_in_batch = (labels[..., 1:] != -100).sum().item()
            input_dict = {
                **input_dict,
                "return_logits": False,
                "num_items_in_batch": num_items_in_batch,
            }
            loss = cast(
                torch.Tensor,
                model.forward_backward(
                    input_kwargs=input_dict,
                    loss_fn=lambda output, _: scaler.scale(output.loss),
                ),
            )
            losses.append(loss.item() / scaler.get_scale())
            model.step(
                lambda: (scaler.step(optim), optim.zero_grad(), None)[-1],
                is_async=is_async,
            )
            scaler.update()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS[is_async])]
    assert sum(diff) / len(diff) < 0.01


def test_qwen3_fused_accumulate():
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=1),
        use_sequential_preset=True,
        optim_dtype=torch.float32,
    )

    optim = Adam(model.optim_parameters(), lr=1e-5)
    scaler = GradScaler(8192.0)
    dataloader = get_dataloader()

    losses = []
    for epoch in range(2):
        for input_dict in dataloader:
            input_ids = input_dict["input_ids"].split(1)
            attention_mask = input_dict["attention_mask"].split(1)
            labels = input_dict["labels"].split(1)
            num_items_in_batch = (input_dict["labels"][..., 1:] != -100).sum().item()
            loss = 0.0
            for mb_input_ids, mb_attention_mask, mb_labels in zip(
                input_ids, attention_mask, labels
            ):
                mb_loss = cast(
                    torch.Tensor,
                    model.forward_backward(
                        input_kwargs={
                            "input_ids": mb_input_ids,
                            "attention_mask": mb_attention_mask,
                            "labels": mb_labels,
                            "return_logits": False,
                            "num_items_in_batch": num_items_in_batch,
                        },
                        loss_fn=lambda output, _: scaler.scale(output.loss),
                    ),
                )
                loss += mb_loss.item()
            losses.append(loss / scaler.get_scale())
            model.step(
                lambda: (scaler.step(optim), optim.zero_grad(), None)[-1],
            )
            scaler.update()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS[True])]
    assert sum(diff) / len(diff) < 0.01
