from typing_extensions import *
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig
from roundpipe.models.function import CompileForCausalLMLoss

MODEL_PATH = os.environ.get("GPT_OSS_20B_PATH", "openai/gpt-oss-20b")
DATASET_PATH = os.environ.get("MATH_500_PATH", "HuggingFaceH4/MATH-500")
REF_LOSS = [3.033435344696045, 2.7854561805725098, 2.7650043964385986, 2.900900363922119, 2.892483711242676, 3.066610336303711, 2.9594414234161377, 2.6337661743164062, 2.7230215072631836, 3.084498882293701] # fmt: skip
REF_AUX_LOSS = [4.004641532897949, 4.004835605621338, 4.003747940063477, 4.004110813140869, 4.005552768707275, 4.0056657791137695, 4.004958629608154, 4.004722595214844, 4.003884792327881, 4.003328800201416] # fmt: skip


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


def test_gpt_oss_auto_forward():
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    raw_model.to(torch.float16)
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


def test_gpt_oss_preset_forward():
    raw_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        dtype=torch.float16,
    )
    raw_model.to(torch.float16)
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
            num_items_in_batch=num_items_in_batch,
        )
        aux_loss = cast(torch.Tensor, out.aux_loss)
        aux_losses.append(aux_loss.item())
        losses.append(
            (
                (cast(torch.Tensor, out.loss) - model.router_aux_loss_coef * aux_loss)
                * 4
            ).item()
        )

    diff = [abs(v - ref) / ref for v, ref in zip(losses, REF_LOSS)]
    assert sum(diff) / len(diff) < 0.005

    aux_diff = [abs(v - ref) / ref for v, ref in zip(aux_losses, REF_AUX_LOSS)]
    assert sum(aux_diff) / len(aux_diff) < 0.01
