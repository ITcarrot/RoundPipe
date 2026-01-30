# Load model directly
from typing_extensions import *
import math

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import time
import sys
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig
from roundpipe.optim import Adam
from roundpipe.models.function import ChunkedCompileLinearForCausalLMLoss

torch.backends.cuda.matmul.allow_fp16_accumulation = True
transformers.modeling_utils._init_weights = False

if len(sys.argv) != 6:
    print(
        f"Usage: python {sys.argv[0]} <model_path> <batch_size> <seq_length> <num_microbatch> <accum_steps>"
    )
    sys.exit(1)
model_path = sys.argv[1]
BS = int(sys.argv[2])
SEQ_LENGTH = int(sys.argv[3])
NUM_MICROBATCH = int(sys.argv[4])
ACCUM_STEPS = int(sys.argv[5])

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
config._attn_implementation = "flash_attention_2"
config.use_cache = False
config.dtype = torch.float16
with torch.device("meta"):
    hf_model = AutoModelForCausalLM.from_config(config)
for module in hf_model.modules():
    module.to_empty(device="cuda", recurse=False)
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    module._apply(lambda t: t.to(device="cpu", non_blocking=True), recurse=False)
torch.cuda.synchronize()
torch.cuda.empty_cache()

model = wrap_model_to_roundpipe(
    hf_model,
    optim_dtype=torch.float32,
    use_sequential_preset=True,
    model_run_config=RoundPipeRunConfig(num_microbatch=NUM_MICROBATCH),
)
optim = Adam(model.optim_parameters(), lr=1e-5)

all_input = torch.randint(0, tokenizer.vocab_size, (BS, SEQ_LENGTH))
input_tensors = all_input.chunk(ACCUM_STEPS)
masks = [torch.ones_like(input_tensor) for input_tensor in input_tensors]

from roundpipe.device import device_list

for device in device_list:
    lm_head = torch.nn.Linear(
        model.lm_head.in_features,
        model.lm_head.out_features,
        model.lm_head.bias is not None,
        device=device.device,
        dtype=torch.float16,
    )
    for microbs in (math.floor(BS / ACCUM_STEPS), math.ceil(BS / ACCUM_STEPS)):
        for minibs in (
            math.floor(microbs / NUM_MICROBATCH),
            math.ceil(microbs / NUM_MICROBATCH),
        ):
            sample_hidden = torch.rand(
                (minibs, SEQ_LENGTH, lm_head.in_features),
                dtype=torch.float16,
                device=device.device,
                requires_grad=True,
            )
            sample_labels = torch.randint(
                0,
                tokenizer.vocab_size,
                (minibs, SEQ_LENGTH),
                dtype=torch.long,
                device=device.device,
            )
            with torch.autocast(
                "cpu", enabled=False, dtype=torch.float16, cache_enabled=False
            ), torch.autocast(
                "cuda", enabled=False, dtype=torch.float16, cache_enabled=False
            ):
                ChunkedCompileLinearForCausalLMLoss(
                    sample_hidden,
                    lm_head,
                    sample_labels,
                )
            del sample_hidden, sample_labels
    del lm_head


def train_iter():
    for input_tensor, mask in zip(input_tensors, masks):
        loss = model.forward_backward(
            input_kwargs={
                "input_ids": input_tensor,
                "attention_mask": mask,
                "labels": input_tensor,
                "return_logits": False,
            },
            loss_fn=lambda outputs, _: outputs.loss / NUM_MICROBATCH / ACCUM_STEPS,
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
