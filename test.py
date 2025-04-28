from accelerate.utils import offload_weight
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(torch.version.cuda)  # This gives the version of CUDA PyTorch was built with
print(torch.backends.cudnn.version())  # (optional) cuDNN version
print(torch.cuda.is_available())  # Checks if CUDA is available on your system
print(torch.cuda.get_device_name(0))  # (if available) Name of the GPU

# model_id = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
# file_name = "Meta-Llama-3-8B-Instruct.Q6_K.gguf"
# model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# file_name = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Enable nested quantization
    bnb_4bit_quant_type="nf4"       # Use Normal Float 4 data type
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    quantization_config=bnb_config
)
print("4-bit Quantization with BitsAndBytes Complete")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
