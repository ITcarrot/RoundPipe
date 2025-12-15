# Load model directly
import torch
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from tqdm import tqdm
import time
from RoundPipe import wrap_model_to_roundpipe, RoundPipe

tokenizer = Qwen2Tokenizer.from_pretrained("/public/huggingface-models/Qwen/Qwen3-8B")
model = Qwen3ForCausalLM.from_pretrained(
    "/public/huggingface-models/Qwen/Qwen3-8B",
    use_cache=False,
)
model: RoundPipe = wrap_model_to_roundpipe(model)
optim = torch.optim.Adam(model.optim_parameters(), lr=1e-5, fused=True)

BS = 10 * (torch.cuda.device_count() + 1)
input_tensor = torch.randint(0, tokenizer.vocab_size, (BS, 1024))
mask = torch.ones_like(input_tensor)

with torch.no_grad():
    loss = model(input_ids=input_tensor, attention_mask=mask, labels=input_tensor).loss
    print(f'Initial loss: {loss.item()}')

torch.cuda.memory._record_memory_history()
for iter in tqdm(range(2)):
    loss = model.forward_backward(input_kwargs={'input_ids': input_tensor, 'attention_mask': mask}, label=input_tensor,
                    loss_fn=lambda outputs, labels: model.loss_function(logits=outputs.logits, labels=labels, vocab_size=model.vocab_size) / (torch.cuda.device_count() + 1))
    model.step(lambda: (optim.step(), optim.zero_grad(), None)[-1])
torch.cuda.memory._dump_snapshot('./my_snapshot.pickle')

start = time.perf_counter()
for iter in tqdm(range(5)):
    loss = model.forward_backward(input_kwargs={'input_ids': input_tensor, 'attention_mask': mask}, label=input_tensor,
                    loss_fn=lambda outputs, labels: model.loss_function(logits=outputs.logits, labels=labels, vocab_size=model.vocab_size) / (torch.cuda.device_count() + 1))
    model.step(lambda: (optim.step(), optim.zero_grad(), None)[-1])

end = time.perf_counter()
print(f"Time taken for 5 iterations: {end - start} seconds")
print(f'Thoughput: {5 * input_tensor.shape[0] * input_tensor.shape[1] / (end - start)} tokens/second')
