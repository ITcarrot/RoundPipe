from typing_extensions import *
import os
import itertools

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss
from datasets import load_dataset
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig, GradScaler
from roundpipe.optim import Adam

MODEL_PATH = (
    os.environ["LLAMA_3_2_1B_PATH"]
    if "LLAMA_3_2_1B_PATH" in os.environ
    else "meta-llama/Llama-3.2-1B"
)
DATASET_PATH = (
    os.environ["MATH_500_PATH"]
    if "MATH_500_PATH" in os.environ
    else "HuggingFaceH4/MATH-500"
)
REF_LOSS = [1.4427820444107056, 1.6373329162597656, 1.4397658109664917, 0.9818760752677917, 1.2102022171020508, 1.1305278539657593, 1.1756712198257446, 1.0232906341552734, 1.2296929359436035, 1.2951247692108154, 0.7375290989875793, 0.9363015294075012, 0.8175673484802246, 0.7015732526779175, 0.7806639075279236, 0.7206956148147583, 0.8156185746192932, 0.6696286797523499, 0.8242655992507935, 0.9606342911720276]  # fmt: skip


def get_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    raw_dataset = load_dataset(DATASET_PATH, split="test")

    samples = []
    for example in raw_dataset:
        example = cast(Dict[str, str], example)
        conversation = (
            f"<|user|>{example['problem']}<|assistant|>{example['solution']}<|end|>"
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


def test_llama3_classic():
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    model = wrap_model_to_roundpipe(
        raw_model,
        model_run_config=RoundPipeRunConfig(num_microbatch=3),
        use_sequential_preset=False,
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
            )
            scaler.update()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS)]
    assert sum(diff) / len(diff) < 0.01


def test_llama3_fused():
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
            num_items_in_batch = (labels[..., 1:] != -100).sum().item()
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
            model.step(lambda: (scaler.step(optim), optim.zero_grad(), None)[-1])
            scaler.update()

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS)]
    assert sum(diff) / len(diff) < 0.01
