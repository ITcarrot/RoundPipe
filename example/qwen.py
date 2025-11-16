# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from datasets import load_dataset
from RoundPipe import wrap_model_to_roundpipe, RoundPipeRunConfig

tokenizer = Qwen2Tokenizer.from_pretrained("/public/huggingface-models/Qwen/Qwen3-0.6B")
model = Qwen3ForCausalLM.from_pretrained(
    "/public/huggingface-models/Qwen/Qwen3-0.6B",
    use_cache=False,
)
model = wrap_model_to_roundpipe(model, model_run_config=RoundPipeRunConfig(num_microbatch=4))
optim = torch.optim.Adam(model.parameters(), lr=1e-5)
scaler = torch.GradScaler()
dataset = load_dataset("/data/lyb/AI-MO/NuminaMath-CoT")

dataset = dataset['train']
text_input = []
for sample in dataset:
    messages = sample["messages"]
    input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if len(input) < 1000:
        text_input.append(input)
    if len(text_input) >= 100:
        break

dataloader = torch.utils.data.DataLoader(text_input, batch_size=4, shuffle=False)
def tokenize(text):
    input_dict = tokenizer(text, return_tensors="pt", padding=True, truncation=False)
    input_ids = input_dict["input_ids"]
    attention_mask = input_dict["attention_mask"]
    labels = input_dict["input_ids"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

losses = []
for epoch in range(20):
    epoch_loss = []
    for data in dataloader:
        input_dict = tokenize(data)
        loss = model(**input_dict).loss
        print(loss.item())
        epoch_loss.append(loss.item())
        loss.backward()
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        optim.step()
        optim.zero_grad()
    mean_loss = sum(epoch_loss) / len(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {mean_loss}")
