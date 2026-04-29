# 微调 Qwen3-1.7B

!!! tip "交互式 Notebook"
    自己动手试试: [qwen3_nekoqa.zh.ipynb](https://github.com/ITcarrot/RoundPipe/blob/main/example/qwen3_nekoqa.zh.ipynb)

本教程演示如何使用 **RoundPipe** 在猫娘对话 [liumindmind/NekoQA-10K](https://huggingface.co/datasets/liumindmind/NekoQA-10K) 数据集上全参微调 Qwen3-1.7B，训练完成后展示模型的生成效果。

你将学到：

1. 如何用 `wrap_model_to_roundpipe()` 一行把 HuggingFace 模型转成 RoundPipe 流水线；
2. 如何搭建大语言模型的微调训练循环（`forward_backward` + `GradScaler` + 异步优化器）；
3. 如何在训练结束后用 `synchronize()` 同步参数并进行推理生成。

> **硬件参考**：Qwen3-1.7B FP16 全参微调约占 34 GB 内存。RoundPipe 将参数放在 CPU 并按需流水加载到 GPU，单张 24 GB 消费卡即可完成全参微调。多卡环境无需改代码，吞吐线性增长。

## 1. 环境与依赖

需要安装：`roundpipe`、`transformers`、`datasets`。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig, GradScaler
from roundpipe.optim import Adam
```

## 2. 加载模型与分词器

- 直接用 HuggingFace Hub ID 加载，也可以换成本地路径；
- `use_cache=False`：RoundPipe 不需要 KV Cache，关掉可以节省内存；
- `dtype=torch.float16`：模型以 FP16 计算，优化器稍后由 RoundPipe 在 CPU 上用 FP32 维护。

```python
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"  # 替换为实际模型权重路径

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    dtype=torch.float16,
)
```

## 3. 用 RoundPipe 包装模型

`wrap_model_to_roundpipe()` 会自动查找适配 Qwen3 的 preset，将模型拆分成 `Prefix -> DecoderLayers -> Postfix` 的 `nn.Sequential`，并交给 RoundPipe 管理。

- `num_microbatch=10`：每个 batch 切成 10 个 microbatch 交错执行，减小显存峰值；
- `optim_dtype=torch.float32`：优化器参数以 FP32 精度保存在 CPU 上。

```python
model = wrap_model_to_roundpipe(
    raw_model,
    model_run_config=RoundPipeRunConfig(num_microbatch=10),
    optim_dtype=torch.float32,
)
optim = Adam(model.optim_parameters(), lr=1e-5)
scaler = GradScaler()
```

## 4. 准备 NekoQA-10K 数据集

数据集的每条样本包含 `instruction`（用户问题）和 `output`（猫娘风格的回复）。

我们用 Qwen3 的 chat template 将它格式化为多轮对话格式，并只取长度适中的样本以加快训练。

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
    if len(text) < 2000:  # 过滤过长样本
        text_input.append(text)

print(f"训练样本数: {len(text_input)}")
print(f"\n样本示例:\n{text_input[0][:300]}...")
```

??? example "Output"

    ```
    训练样本数: 10041

    样本示例:
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
    """将文本 batch 分词并构造 labels（pad 位置设为 -100 跳过 loss 计算）。"""
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, labels
```

## 5. 微调前的基线输出

在训练之前，我们先看看原始 Qwen3-1.7B-Base 对猫娘风格问题的回复。由于这是基座模型，它就是一个复读机。

> 下面的 `generate()` 函数通过手动自回归循环实现生成（因为 RoundPipe 不支持 KV Cache），生成速度较慢仅用于演示。

```python
@torch.no_grad()
def generate(prompt, max_new_tokens=128):
    """简单的贪心解码，用于微调前后的效果对比。"""
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
    "早上好！",
    "今天天气真好，我们出去玩好不好？",
    "读研好累，不想读了……",
]

print("===== 微调前 =====")
for p in test_prompts:
    print(f"\n用户: {p}")
    print(f"模型: {generate(p)}")
```

??? example "Output"

    ```
    ===== 微调前 =====

    用户: 早上好！
    模型: user
    早上好！
    assistant
    早上好！תשוב
    תשובassistant
    早上好！תשובassistant
    早上好！תשובassistant
    ...

    用户: 今天天气真好，我们出去玩好不好？
    模型: user
    今天天气真好，我们出去玩好不好？
    assistant
    好啊，我们出去玩吧。我们去公园吧。我们去公园吧。我们去公园吧。
    我们去公园吧。我们去公园吧。我们去公园吧。我们去公园吧。...

    用户: 读研好累，不想读了……
    模型: user
    读研好累，不想读了……
    assistant
    Assistant: 你看起来有些疲惫，是不是遇到了什么困难？如果你觉得读研太累，
    可以考虑调整一下自己的学习计划，或者寻求一些帮助和支持。...
    ```

## 6. 训练循环

使用 RoundPipe 的 `forward_backward()` API：

- `input_kwargs`：传入 `input_ids` 和 `attention_mask`；
- `label`：传入 `labels`，Qwen3 preset 会在 Postfix 层自动计算交叉熵损失；
- `loss_fn`：在这里用 `scaler.scale(...)` 做梯度缩放，并除以 microbatch 数量做平均；
- `model.step()`：异步执行优化器更新（梯度反缩放 -> 优化器 step -> 梯度清零）。

```python
NUM_EPOCHS = 2
NUM_MICROBATCH = 10  # 和 RoundPipeRunConfig 一致

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
    Epoch 1/2, Loss: 1.9918
    Epoch 2/2, Loss: 1.5527
    ```

## 7. 微调后的输出

训练结束后，用同样的测试问题看看模型是否学会了猫娘的说话风格。

```python
print("===== 微调后 =====")
for p in test_prompts:
    print(f"\n用户: {p}")
    print(f"模型: {generate(p)}")
```

??? example "Output"

    ```
    ===== 微调后 =====

    用户: 早上好！
    模型: user
    早上好！
    assistant
    <think>

    </think>

    喵~主人早呀！(眼睛闪闪发光，尾巴轻轻摇晃)今天也要加油哦！要不要摸摸我的头？

    用户: 今天天气真好，我们出去玩好不好？
    模型: user
    今天天气真好，我们出去玩好不好？
    assistant
    <think>

    </think>

    喵~主人今天心情这么好呀！(眼睛闪闪发光)宝宝也好想出去玩的说！
    外面有好多好玩的东西喵~我们可以去公园散步，或者去河边玩水！
    (摇尾巴)主人想带宝宝去什么地方呀？宝宝最喜欢和主人一起玩耍了喵~

    用户: 读研好累，不想读了……
    模型: user
    读研好累，不想读了……
    assistant
    <think>

    </think>

    呜喵~主人不要这么说嘛！(轻轻蹭蹭主人的手)宝宝知道主人很辛苦的说...
    但是读研虽然累，但是能学到好多好多知识，对主人来说一定是很重要的事情喵~
    (歪着头，耳朵轻轻抖动)

    主人要不要先休息一下？宝宝可以给主人揉揉肩膀，或者陪主人一起窝在沙发上看电视喵~
    虽然宝宝可能不太懂那些复杂的课程，但是能陪在主人身边就很开心了的说！
    (摇尾巴，眼睛闪闪发光)

    主人要是累了，就摸摸宝宝的头吧~这样心情会...
    ```

## 8. 保存微调后的模型

调用 `model.synchronize()` 同步参数后，微调后的模型参数均位于原位，直接保存 `raw_model` 或调用 `model.save_pretrained()` 均可。

RoundPipe 会转发所有对 `model` 调用到原来的类上，因此可以用 HuggingFace 模型 API 直接操作 `model`。

```python
model.synchronize()
model.save_pretrained("./Qwen3-1.7B-nekoqa-finetuned")
```

## 小结

本教程展示了使用 RoundPipe 微调大语言模型的完整流程：

1. **一行包装**：`wrap_model_to_roundpipe()` 自动适配 Qwen3 架构，不需要手动拆分层。
2. **融合训练 API**：`forward_backward()` 将前向和反向融合，配合 `GradScaler` 做 FP16 混合精度训练，`model.step()` 异步更新优化器。
3. **推理**：训练后调用 `model.synchronize()` 同步参数，再切换到 `eval()` 模式即可做推理。
4. **保存**：调用 `model.synchronize()` 后，直接使用 `model.save_pretrained()` 或保存 `raw_model` 即可导出微调结果。

**进一步探索：**

- 调大 `num_microbatch` 可以减小显存峰值，适配更长的序列；
- 多 GPU 环境无需改代码，RoundPipe 自动分配流水段。
