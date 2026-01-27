# Load model directly
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, TaskType, LoraModel
from tqdm import tqdm
import time
import sys
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig
from roundpipe.optim import Adam

torch._dynamo.config.recompile_limit = 2
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
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = LoraModel(hf_model, lora_config, "default").model
for module in lora_model.modules():
    module.to_empty(device="cuda", recurse=False)
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    module._apply(lambda t: t.to(dtype=torch.float16, device="cpu"), recurse=False)
torch.cuda.synchronize()
torch.cuda.empty_cache()

model = wrap_model_to_roundpipe(
    lora_model,
    optim_dtype=torch.float32,
    use_sequential_preset=True,
    model_run_config=RoundPipeRunConfig(num_microbatch=NUM_MICROBATCH),
    pin_with_register=True,
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
