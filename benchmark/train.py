# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import sys
from roundpipe import wrap_model_to_roundpipe, RoundPipe
from roundpipe.optim import Adam

torch.backends.cuda.matmul.allow_fp16_accumulation = True

if len(sys.argv) != 5:
    print(
        f"Usage: python {sys.argv[0]} <model_path> <batch_size> <seq_length> <accum_steps>"
    )
    sys.exit(1)
model_path = sys.argv[1]
BS = int(sys.argv[2])
SEQ_LENGTH = int(sys.argv[3])
ACCUM_STEPS = int(sys.argv[4])

tokenizer = AutoTokenizer.from_pretrained(model_path)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    _attn_implementation="flash_attention_2",
    use_cache=False,
    dtype=torch.float16,
)
model = wrap_model_to_roundpipe(
    hf_model, optim_dtype=torch.float32, use_sequential_preset=True
)
optim = Adam(model.optim_parameters(), lr=1e-5)

all_input = torch.randint(0, tokenizer.vocab_size, (BS, SEQ_LENGTH))
input_tensors = all_input.chunk(ACCUM_STEPS)
masks = [torch.ones_like(input_tensor) for input_tensor in input_tensors]


def train_iter():
    for input_tensor, mask in zip(input_tensors, masks):
        loss = model.forward_backward(
            input_kwargs={"input_ids": input_tensor, "attention_mask": mask},
            label=input_tensor,
            loss_fn=lambda outputs, labels: model.loss_function(
                logits=outputs.logits, labels=labels, vocab_size=model.vocab_size
            )
            / (torch.cuda.device_count() + 1),
        )
    model.step(lambda: (optim.step(), optim.zero_grad(), None)[-1])


for iter in tqdm(range(5)):
    train_iter()
start = time.perf_counter()
for iter in tqdm(range(10)):
    train_iter()
end = time.perf_counter()
print(f"Time taken for 10 iterations: {end - start} seconds")
print(
    f"Thoughput: {10 * all_input.shape[0] * all_input.shape[1] / (end - start)} tokens/second"
)
