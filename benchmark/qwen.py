# Load model directly
import torch
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from tqdm import tqdm
import time
from roundpipe import wrap_model_to_roundpipe, RoundPipe, GradScaler, RoundPipeRunConfig

torch.backends.cuda.matmul.allow_fp16_accumulation = True

tokenizer = Qwen2Tokenizer.from_pretrained("/public/huggingface-models/Qwen/Qwen3-8B")
model = Qwen3ForCausalLM.from_pretrained(
    "/public/huggingface-models/Qwen/Qwen3-0.6B",
    _attn_implementation="flash_attention_2",
    use_cache=False,
    dtype=torch.float16,
)
BS = 16 * 8
config = RoundPipeRunConfig(num_microbatch=8)
model = wrap_model_to_roundpipe(
    model, model_run_config=config, optim_dtype=torch.float32
)
optim = torch.optim.Adam(model.optim_parameters(), lr=1e-5, fused=True)
scaler = GradScaler(4096.0)

base_text = (
    "This is a meaningful natural language sentence used for benchmarking "
    "training speed, numerical stability, and forward backward throughput. "
)
tokens = []
while len(tokens) < 1024:
    tokens.extend(tokenizer.encode(base_text, add_special_tokens=False))
tokens = tokens[:1024]
input_tensor = torch.tensor([tokens] * BS)
mask = torch.ones_like(input_tensor)

with torch.no_grad():
    loss = model(input_ids=input_tensor, attention_mask=mask, labels=input_tensor).loss
    print(f"Initial loss: {loss.item()}")

# torch.cuda.memory._record_memory_history()
for iter in tqdm(range(2)):
    loss = model.forward_backward(
        input_kwargs={"input_ids": input_tensor, "attention_mask": mask},
        label=input_tensor,
        loss_fn=lambda outputs, labels: scaler.scale(
            model.loss_function(
                logits=outputs.logits, labels=labels, vocab_size=model.vocab_size
            )
            / (torch.cuda.device_count() + 1)
        ),
    )
    print(loss.item(), scaler.get_scale())
    model.step(lambda: (scaler.step(optim), optim.zero_grad(), None)[-1])
    scaler.update()
# torch.cuda.memory._dump_snapshot('./my_snapshot.pickle')

start = time.perf_counter()
for iter in tqdm(range(5)):
    loss = model.forward_backward(
        input_kwargs={"input_ids": input_tensor, "attention_mask": mask},
        label=input_tensor,
        loss_fn=lambda outputs, labels: scaler.scale(
            model.loss_function(
                logits=outputs.logits, labels=labels, vocab_size=model.vocab_size
            )
            / (torch.cuda.device_count() + 1)
        ),
    )
    print(loss.item(), scaler.get_scale())
    model.step(lambda: (scaler.step(optim), optim.zero_grad(), None)[-1])
    scaler.update()

end = time.perf_counter()
print(f"Time taken for 5 iterations: {end - start} seconds")
print(
    f"Thoughput: {5 * input_tensor.shape[0] * input_tensor.shape[1] / (end - start)} tokens/second"
)
