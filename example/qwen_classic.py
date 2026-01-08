# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from datasets import load_dataset
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig, GradScaler

tokenizer = Qwen2Tokenizer.from_pretrained("/public/huggingface-models/Qwen/Qwen3-0.6B")
model = Qwen3ForCausalLM.from_pretrained(
    "/public/huggingface-models/Qwen/Qwen3-0.6B",
    _attn_implementation="flash_attention_2",
    use_cache=False,
    dtype=torch.float16,
)
model = wrap_model_to_roundpipe(
    model,
    model_run_config=RoundPipeRunConfig(num_microbatch=4),
    use_sequential_preset=False,
    optim_dtype=torch.float32,
)
optim = torch.optim.Adam(model.optim_parameters(), lr=1e-5, fused=True)
scaler = GradScaler(8192.0)
dataset = load_dataset("/data/lyb/AI-MO/NuminaMath-CoT")

dataset = dataset["train"]
text_input = []
for sample in dataset:
    messages = sample["messages"]
    input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if len(input) < 1000:
        text_input.append(input)
    if len(text_input) >= 100:
        break

dataloader = torch.utils.data.DataLoader(text_input, batch_size=4, shuffle=False)


def tokenize(text):
    input_dict = tokenizer(text, return_tensors="pt", padding=True, truncation=False)
    input_ids = input_dict["input_ids"]
    attention_mask = input_dict["attention_mask"]
    labels = input_dict["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


losses = []
for epoch in range(20):
    epoch_loss = []
    for data in dataloader:
        input_dict = tokenize(data)
        loss = model(**input_dict).loss
        scaler.scale(loss).backward()
        print(loss.item(), scaler.get_scale())
        epoch_loss.append(loss.item())
        model.step(lambda: (scaler.step(optim), optim.zero_grad(), None)[-1])
        scaler.update()
    mean_loss = sum(epoch_loss) / len(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {mean_loss}")
