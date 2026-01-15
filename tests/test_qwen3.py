from typing_extensions import *
import os
import itertools
import pathlib
import pickle

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss
from datasets import load_dataset
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig, GradScaler
from roundpipe.optim import Adam

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
            loss = ForCausalLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=model.config.vocab_size,
            )
            loss.backward()
            losses.append(loss.item())
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            optim.step()
            optim.zero_grad()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS[False])]
    assert sum(diff) / len(diff) < 0.001


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
            loss = ForCausalLMLoss(
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
            labels = input_dict.pop("labels")
            num_items_in_batch = (labels != -100).sum().item()
            loss = cast(
                torch.Tensor,
                model.forward_backward(
                    input_kwargs=dict(input_dict),
                    label=labels,
                    loss_fn=lambda outputs, labels: scaler.scale(
                        model.loss_function(
                            logits=outputs.logits,
                            labels=labels,
                            vocab_size=model.vocab_size,
                            num_items_in_batch=num_items_in_batch,
                        )
                    ),
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
