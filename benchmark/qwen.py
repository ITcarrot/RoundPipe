# Load model directly
import torch
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from tqdm import tqdm
import time
from RoundPipe import wrap_model_to_roundpipe

tokenizer = Qwen2Tokenizer.from_pretrained("/public/huggingface-models/Qwen/Qwen3-8B")
model = Qwen3ForCausalLM.from_pretrained(
    "/public/huggingface-models/Qwen/Qwen3-8B",
    use_cache=False,
)
model = wrap_model_to_roundpipe(model)
optim = torch.optim.Adam(model.parameters(), lr=1e-5, fused=True)

BS = 2 * (torch.cuda.device_count() + 1)
input_tensor = torch.randint(0, tokenizer.vocab_size, (BS, 1024))
mask = torch.ones_like(input_tensor)

with torch.no_grad():
    loss = model(input_ids=input_tensor, attention_mask=mask, labels=input_tensor).loss
    print(f'Initial loss: {loss.item()}')

for iter in tqdm(range(2)):
    loss, _ = model.train_iter(input_kwargs={'input_ids': input_tensor, 'attention_mask': mask}, label=input_tensor,
                    loss_fn=lambda outputs, labels: model.loss_function(logits=outputs.logits, labels=labels, vocab_size=model.vocab_size) / (torch.cuda.device_count() + 1))
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    optim.step()
    optim.zero_grad()

start = time.perf_counter()
for iter in tqdm(range(5)):
    loss, _ = model.train_iter(input_kwargs={'input_ids': input_tensor, 'attention_mask': mask}, label=input_tensor,
                    loss_fn=lambda outputs, labels: model.loss_function(logits=outputs.logits, labels=labels, vocab_size=model.vocab_size) / (torch.cuda.device_count() + 1))
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    optim.step()
    optim.zero_grad()

end = time.perf_counter()
print(f"Time taken for 5 iterations: {end - start} seconds")
print(f'Thoughput: {5 * input_tensor.shape[0] * input_tensor.shape[1] / (end - start)} tokens/second')
