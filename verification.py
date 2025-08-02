import numpy as np
from accelerate.utils import offload_weight
from transformers import AutoTokenizer, AutoModelForCausalLM
from context_cite import ContextCiter
import json
import random
from tqdm import tqdm
import torch
from attributor import Attributor
from time import time
import re
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())

# model_id = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
# file_name = "Meta-Llama-3-8B-Instruct.Q6_K.gguf"
# model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# file_name = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
from transformers import BitsAndBytesConfig

def top_k_binary_vector(arr, k):
    arr = np.array(arr)
    if k >= len(arr):
        return np.ones_like(arr, dtype=int)
    # Get indices of the top k values
    top_k_indices = np.argpartition(-arr, k)[:k]
    # Create a binary vector
    binary_vector = np.zeros_like(arr, dtype=bool)
    binary_vector[top_k_indices] = 1
    return binary_vector

def normalize_input(s):
    # Remove special tokens like <|eot_id|>
    s = re.sub(r'<\|.*?\|>', '', s)
    # Lowercase, strip whitespace, remove punctuation
    s = re.sub(r'[^\w\s]', '', s.strip().lower())
    return s

def generate_response(model, tokenizer, prompt):
    chat_promp = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    print("Chat prompt:", chat_promp)
    input_ids = tokenizer.encode(chat_promp, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    response_ids = output_ids[:, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids[0], add_special_tokens=False)
    return response

def clean_merged_statement(text):
    pattern = r'^here is the merged statement:\s*'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Enable nested quantization
    bnb_4bit_quant_type="nf4"       # Use Normal Float 4 data type
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="cuda",
    quantization_config=bnb_config
)
model.to("cuda")
print("4-bit Quantization with BitsAndBytes Complete")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Input and output file paths
input_file = '../NQ_dataset/nq-simple.jsonl'
# TODO: set to 1000
num_samples = 1000
random.seed(42)
max_context_length = 15000

with open(input_file, 'r', encoding='utf-8') as fin:
    num_lines = sum(1 for _ in fin)
sample_indices = sorted(random.sample(range(num_lines), num_samples))

i_line = 0
i_index = 0
ks = [1, 2, 4, 8, 16]
# counter of yes for 1, 2, 4, 8, 16, all ks
yes_counter = [0, 0, 0, 0, 0, 0]
with open(input_file, 'r', encoding='utf-8') as fin:
    for line in tqdm(fin, desc="Processing"):
        if i_line != sample_indices[i_index]:
            i_line += 1
            continue
        data = json.loads(line)

        query = data.get("question", "")
        context = data.get("context", "")
        if len(context) > max_context_length:
            i_line += 1
            sample_indices[i_index] += 1
            print("\n\n\n SKIPPED\n\n\n\n\n")
            continue
        print("Line", i_index, f"({i_line}/{num_lines})")
        # TODO: set num_ablations to 256
        attributor = Attributor(model, tokenizer, context, query, device="cuda", batch_size=1, num_ablations=256)
        attributor.prompt_template += "\n\nPlease answer with a single word or phrase when possible.\nIf the question cannot be answered from the context, say so instead.\n"
        print("Context:", context)
        print("Question:", query)
        print("Answer:", attributor.get_response())
        start = time()
        result = attributor.get_attributions()
        end = time()
        print("Most important source:", attributor.context_split[np.argmax(result)])
        print("Score:", np.max(result))
        print("Time:", end - start)
        verification_prompt = [{"role": "user", "content": f"""Please merge the following question and answer into a single statement. For
example, if the question is "What is the capital of France?" and the answer is
"Paris", you should say: "The capital of France is Paris.
Question: {query}
Answer: {attributor.get_response()}
"""}]
        verification_response = generate_response(model, tokenizer, verification_prompt)
        verification_response = clean_merged_statement(verification_response)
        print("Verification response:", verification_response)
        print(result)
        for i, k in enumerate(ks):
            print(f"\nK {k} --------------------------------------------------------------------------------------------------------------------------------------- \n")
            ablation_vector = top_k_binary_vector(result, k)
            print(ablation_vector)
            ablated_context = attributor.create_ablated_context(ablation_vector)
            ablated_prompt = [{"role": "user", "content": f"Context: {ablated_context}\n\nCan we conclude that \"{verification_response}\"? Please respond with just yes or no.\n"}]

            ablated_response = generate_response(model, tokenizer, ablated_prompt)
            print("Conclusion:", ablated_response)
            normalized_response = normalize_input(ablated_response)
            if normalized_response == "yes":
                yes_counter[i] += 1
            elif normalized_response != "no":
                raise Exception(f"Unexpected response: {normalized_response}")
        print("\nK all --------------------------------------------------------------------------------------------------------------------------------------- \n")
        ablated_prompt = [{"role": "user",
                           "content": f"Context: {context}\n\nCan we conclude that \"{verification_response}\"? Please respond with just yes or no.\n"}]
        ablated_response = generate_response(model, tokenizer, ablated_prompt)
        print("Conclusion:", ablated_response)
        normalized_response = normalize_input(ablated_response)
        if normalized_response == "yes":
            yes_counter[len(ks)] += 1
        elif normalized_response != "no":
            raise Exception(f"Unexpected response: {normalized_response}")
        print("YES COUNTER", yes_counter, "\n\n\n\n\n\n\n\n")
        i_line += 1
        i_index += 1
        if i_index >= num_samples:
            break

print("TOTAL TEST SAMPLES:", num_samples)

print("YES COUNTER:", yes_counter)
