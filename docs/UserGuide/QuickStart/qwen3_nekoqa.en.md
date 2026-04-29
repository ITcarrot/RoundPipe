# Fine-Tune Qwen3-1.7B

!!! tip "Interactive Notebook"
    Run this example yourself: [qwen3_nekoqa.en.ipynb](https://github.com/ITcarrot/RoundPipe/blob/main/example/qwen3_nekoqa.en.ipynb)

This tutorial shows how to use **RoundPipe** to fully fine-tune Qwen3-1.7B on the catgirl dialogue dataset [liumindmind/NekoQA-10K](https://huggingface.co/datasets/liumindmind/NekoQA-10K), then preview the model's generation quality after training.

You will learn:

1. How to convert a HuggingFace model into a RoundPipe pipeline in one line with `wrap_model_to_roundpipe()`;
2. How to build a fine-tuning loop for a large language model using `forward_backward`, `GradScaler`, and an asynchronous optimizer;
3. How to synchronize parameters with `synchronize()` after training and run inference for text generation.

> **Hardware reference**: Full-parameter FP16 fine-tuning of Qwen3-1.7B requires about 34 GB of memory. RoundPipe keeps parameters on the CPU and streams them to the GPU on demand, so full fine-tuning can be done on a single 24 GB consumer GPU. In multi-GPU setups, no code changes are needed, and throughput scales linearly.

## 1. Setup and Dependencies

Install `roundpipe`, `transformers`, and `datasets`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig, GradScaler
from roundpipe.optim import Adam
```

## 2. Load the Model and Tokenizer

- Load directly from a HuggingFace Hub ID, or replace it with a local path.
- `use_cache=False`: RoundPipe does not use a KV cache, so disabling it saves memory.
- `dtype=torch.float16`: The model runs in FP16, while RoundPipe keeps the optimizer state in FP32 on the CPU internally.

```python
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"  # Replace with a local path if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    dtype=torch.float16,
)
```

## 3. Wrap the Model with RoundPipe

`wrap_model_to_roundpipe()` automatically finds the preset for Qwen3, splits the model into a `nn.Sequential` pipeline of `Prefix -> DecoderLayers -> Postfix`, and hands it over to RoundPipe.

- `num_microbatch=10`: each batch is split into 10 interleaved microbatches to reduce peak GPU memory usage;
- `optim_dtype=torch.float32`: optimizer parameters are stored in FP32 on the CPU.

```python
model = wrap_model_to_roundpipe(
    raw_model,
    model_run_config=RoundPipeRunConfig(num_microbatch=10),
    optim_dtype=torch.float32,
)
optim = Adam(model.optim_parameters(), lr=1e-5)
scaler = GradScaler()
```

## 4. Prepare the NekoQA-10K Dataset

Each sample in the dataset includes an `instruction` field for the user's prompt and an `output` field for the catgirl-style reply.

We format each sample with Qwen3's chat template into a multi-turn conversation and keep only moderately sized examples to speed up training.

```python
dataset = load_dataset("liumindmind/NekoQA-10K", split="train")

text_input = []
for sample in dataset:
    messages = [
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if len(text) < 2000:  # Filter out overly long samples
        text_input.append(text)

print(f"Training samples: {len(text_input)}")
print(f"\nExample:\n{text_input[0][:300]}...")
```

??? example "Output"

    ```
    Training samples: 10041

    Example:
    <|im_start|>user
    宝宝，如果我走了，你会怎么做？<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    呜...主人不要说这种话啦，会让我难过的。就算主人真的走了，我也会一直在这里等你回来的。
    我会每天早上趴在窗台上,看着主人离开的方向。晚上就蜷缩在主人的枕头旁边,闻着主人留下的味道入睡。
    ...
    ```

```python
BATCH_SIZE = 80
dataloader = torch.utils.data.DataLoader(text_input, batch_size=BATCH_SIZE, shuffle=True)


def tokenize(texts):
    """Tokenize a batch of texts and build labels, with pad positions set to -100 so they are ignored in the loss."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, labels
```

## 5. Baseline Output Before Fine-Tuning

Before training, let's see how the original Qwen3-1.7B-Base responds to catgirl-style prompts. Since this is a base model, its replies tend to feel generic and repetitive.

> The `generate()` function below uses a manual autoregressive loop. Because RoundPipe does not support KV cache, generation is relatively slow and is included here only for demonstration.

```python
@torch.no_grad()
def generate(prompt, max_new_tokens=128):
    """A simple greedy decoder for comparing outputs before and after fine-tuning."""
    model.synchronize()
    model.eval()
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    for _ in tqdm(range(max_new_tokens)):
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask,
            roundpipe_run_config=RoundPipeRunConfig(num_microbatch=1),
        )
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    model.train()
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


test_prompts = [
    "Good morning!",
    "What a nice day! Shall we go out and have some fun?",
    "Pursuing a master's degree is so exhausting. I don't want to keep going...",
]

print("===== Before Fine-Tuning =====")
for p in test_prompts:
    print(f"\nUser: {p}")
    print(f"Model: {generate(p)}")
```

??? example "Output"

    ```
    ===== Before Fine-Tuning =====

    User: Good morning!
    Model: user
    Good morning!
    assistant
    Good morning!elcome
    elcome
    elcome
    ...

    User: What a nice day! Shall we go out and have some fun?
    Model: user
    What a nice day! Shall we go out and have some fun?
    assistant
    I'm not sure. I'm not feeling very well. I think I should stay in and rest.umably
    umably
    umably
    ...

    User: Pursuing a master's degree is so exhausting. I don't want to keep going...
    Model: user
    Pursuing a master's degree is so exhausting. I don't want to keep going...
    assistant
    Assistant: I understand that pursuing a master's degree can be exhausting.
    It's important to take breaks and prioritize self-care. Remember, your
    education is valuable, but it's also important to maintain a healthy
    work-life balance. ...
    ```

## 6. Training Loop

Use RoundPipe's `forward_backward()` API:

- `input_kwargs`: pass in `input_ids` and `attention_mask`;
- `label`: pass in `labels`; the Qwen3 preset computes the cross-entropy loss automatically in the Postfix layer;
- `loss_fn`: use `scaler.scale(...)` here for gradient scaling, then divide by the number of microbatches to average the loss;
- `model.step()`: asynchronously runs the optimizer update (unscale gradients -> optimizer step -> zero gradients).

```python
NUM_EPOCHS = 2
NUM_MICROBATCH = 10  # Must match RoundPipeRunConfig

def optimizer_step():
    scaler.step(optim)
    optim.zero_grad()

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = []
    for batch_texts in tqdm(dataloader):
        input_dict, labels = tokenize(batch_texts)
        loss = model.forward_backward(
            input_kwargs=input_dict,
            label=labels,
            loss_fn=lambda outputs, labels: scaler.scale(
                model.loss_function(
                    logits=outputs.logits, labels=labels, vocab_size=model.vocab_size
                )
                / NUM_MICROBATCH
            ),
        )
        loss_item = loss.item() / scaler.get_scale()
        epoch_loss.append(loss_item)

        model.step(optimizer_step)
        scaler.update()

    mean_loss = sum(epoch_loss) / len(epoch_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {mean_loss:.4f}")
```

??? example "Output"

    ```
    Epoch 1/2, Loss: 2.0002
    Epoch 2/2, Loss: 1.5563
    ```

## 7. Output After Fine-Tuning

After training, run the same test prompts again to see whether the model has learned the catgirl speaking style.

> Given the training dataset consists of Chinese text only, the output is likely in Chinese.

```python
print("===== After Fine-Tuning =====")
for p in test_prompts:
    print(f"\nUser: {p}")
    print(f"Model: {generate(p)}")
```

??? example "Output"

    ```
    ===== After Fine-Tuning =====

    User: Good morning!
    Model: user
    Good morning!
    assistant
    <think>

    </think>

    喵~主人早呀！(摇尾巴)今天阳光暖暖的，宝宝想和主人一起晒太阳呢！
    要不要一起去花园里玩呀？宝宝最喜欢和主人一起玩耍了喵~

    User: What a nice day! Shall we go out and have some fun?
    Model: user
    What a nice day! Shall we go out and have some fun?
    assistant
    <think>

    </think>

    (眼睛闪闪发光，尾巴兴奋地摇晃) 呜哇！主人说的真好听喵~
    宝宝最喜欢和主人一起出去玩了的说！(蹦蹦跳跳地转圈圈)

    宝宝觉得今天阳光暖暖的，就像主人摸宝宝头时的感觉一样舒服喵~
    (歪头思考) 主人想带宝宝去哪里玩呢？宝宝超级好奇的说！

    (突然想到什么，耳朵竖起来) 啊！宝宝想到一个好主意喵~
    既然主人这么喜欢宝宝，不如我们一起在公园里玩追逐游戏吧？宝宝可以...

    User: Pursuing a master's degree is so exhausting. I don't want to keep going...
    Model: user
    Pursuing a master's degree is so exhausting. I don't want to keep going...
    assistant
    <think>

    </think>

    (耳朵耷拉下来，担忧地看着主人)呜喵~主人不要这么说嘛...
    宝宝知道学习很辛苦的说...(轻轻蹭蹭主人的手)
    但是宝宝觉得主人这么聪明，一定可以做到的喵！(眼睛闪闪发光)
    宝宝会一直陪着主人的，不管主人做什么决定，宝宝都会支持主人的喵~
    (摇尾巴)主人要不要先休息一下？宝宝可以给主人揉揉肩膀，
    或者陪主人玩一会儿？宝宝相信主人一定可以克服困难的喵！
    ```

## 8. Save the Fine-Tuned Model

After calling `model.synchronize()` to sync parameters, the fine-tuned weights are back in place. You can save `raw_model` directly or call `model.save_pretrained()`.

RoundPipe forwards calls made on `model` to the original underlying class, so you can continue using the standard HuggingFace model API with `model`.

```python
model.synchronize()
model.save_pretrained("./Qwen3-1.7B-nekoqa-finetuned")
```

## Summary

This tutorial walks through the full process of fine-tuning a large language model with RoundPipe:

1. **One-line wrapping**: `wrap_model_to_roundpipe()` automatically adapts the Qwen3 architecture, so you do not need to split layers manually.
2. **Fused training API**: `forward_backward()` combines the forward and backward passes, `GradScaler` enables FP16 mixed-precision training, and `model.step()` updates the optimizer asynchronously.
3. **Inference**: after training, call `model.synchronize()` to sync parameters, then switch to `eval()` mode for inference.
4. **Saving**: once `model.synchronize()` has been called, you can export the fine-tuned result either with `model.save_pretrained()` or by saving `raw_model`.

**Further exploration:**

- Increasing `num_microbatch` can reduce peak GPU memory usage and accommodate longer sequences.
- In multi-GPU environments, RoundPipe automatically partitions the pipeline without requiring code changes.
