import random
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as ch
import nltk
from itertools import product
from sklearn.linear_model import Lasso
import numpy as np
from time import time
from transformers import DataCollatorForSeq2Seq

class Attributor:
    def __init__(self, model, tokenizer, context: str, query: str, num_ablations: int = 64, lasso_alpha: float = 0.01, device: str = "cpu", batch_size: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.query = query
        # split the context
        self.context_split = []
        for line in self.context.splitlines():
            self.context_split.extend(nltk.sent_tokenize(line))
        # different datasets might need different templates
        self.prompt_template = "Context: {context}\n\nQuery: {query}"
        self.response = None
        self.response_ids = None
        self.attribution = None
        self.num_ablations = num_ablations
        self.lasso_alpha = lasso_alpha
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(cls, model_name: str, context: str, query: str, num_ablations: int = 64, lasso_alpha: float = 0.01, device: str = "auto", batch_size: int = 1):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        return cls(model, tokenizer, context, query, num_ablations, lasso_alpha, device, batch_size)


    # generate all possible ablation vectors - only suitable for very small data
    def all_ablations(self, length):
        return np.array([list(p) for p in product([0, 1], repeat=length)])

    # return n random ablation vectors
    def get_ablations(self, num_sources, n, prob=0.5):
        return [random.choices([False, True], k=num_sources, cum_weights=[1 - prob, 1]) for _ in range(n)]

    def create_ablated_context(self, ablation_vector):
        ablated_context = ""
        for i in range(len(self.context_split)):
            if ablation_vector[i] == 1:
                if len(ablated_context) > 0:
                    ablated_context += " "
                ablated_context += self.context_split[i]

        return ablated_context

    def generate_response(self):
        prompt = [{"role": "user", "content": self.prompt_template.format(context=self.context, query=self.query)}]
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with ch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=512, pad_token_id=self.tokenizer.pad_token_id)

        self.response_ids = output_ids[:, input_ids.shape[1]:]
        # self.full_text = self.tokenizer.decode(output_ids[0])
        self.response = self.tokenizer.decode(self.response_ids[0], add_special_tokens=False)

    def get_logit(self, prompt):
        start = time()
        with ch.no_grad():
            output = self.model(prompt)
        end = time()
        print("  Time taken to generate prompt: {:.2f}s".format(end - start))
        start = time()
        logits = output.logits
        # print("  -- logits len:", logits.shape)
        # print("  -- prompt ids len:", prompt[0])
        # print("  -- response:", self.response_ids[0])
        # log_probs = ch.nn.functional.log_softmax(logits, dim=-1)

        response_probs = ch.zeros(self.response_ids.shape[1])
        for i in range(self.response_ids.shape[1]):
            token_id = self.response_ids[0, i]
            # print("  ", token_id == prompt[0, prompt.shape[1] - self.response_ids.shape[1] + i].item())
            token_log_prob = logits[0, prompt.shape[1] - self.response_ids.shape[1] + i, token_id].item()
            # print(prompt[0, prompt.shape[1] - self.response_ids.shape[1] + i].item(), end=" ")
            response_probs[i] = token_log_prob
        sum_log_probs = ch.sum(ch.nn.functional.log_softmax(response_probs, dim=0))
        # print("  -- sum log probs", sum_log_probs.item())
        # print("  -- prob", ch.exp(sum_log_probs).item())
        end = time()
        # print("  Logit calculation time", end - start)
        # # change back to logit
        # print("  -- logit:", sum_log_probs.item() - ch.log1p(-ch.exp(sum_log_probs)).item())
        # print("  -- mean:", response_probs.mean().item())
        return sum_log_probs.item() - ch.log1p(-ch.exp(sum_log_probs)).item()


    def get_response(self):
        if self.response is None:
            self.generate_response()
        return self.response

    def get_response_ids(self):
        if self.response_ids is None:
            self.generate_response()
        return self.response_ids


    def get_dataset(self, ablation_vectors):
        dataset = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for mask in ablation_vectors:
            ablated_context = self.create_ablated_context(mask)
            # ablated_context = self.create_ablated_context(ablation)
            ablated_prompt = [
                {"role": "user", "content": self.prompt_template.format(context=ablated_context, query=self.query)}]
            ablated_prompt = self.tokenizer.apply_chat_template(ablated_prompt, tokenize=False,
                                                                add_generation_prompt=True)
            prompt_ids = self.tokenizer.encode(ablated_prompt, return_tensors="pt").to(self.device)
            # print(prompt_ids)
            # print(prompt_ids.shape)
            # print(self.response_ids)
            # print(self.response_ids.tolist())
            # ablated_ids = ch.cat((prompt_ids, self.response_ids), dim=1)
            ablated_ids = prompt_ids.tolist()[0] + self.response_ids.tolist()[0]
            dataset["input_ids"].append(ablated_ids)
            dataset["attention_mask"].append([1] * len(ablated_ids))
            dataset["labels"].append([-100] * prompt_ids.shape[1] + self.response_ids.tolist()[0])
        # print(dataset)
        return Dataset.from_dict(dataset)


    def get_attributions(self):
        """
        Finds which parts of context attribute the most to the response generation

        :param num_ablations: Number of ablations to learn a sparse linear surrogate model of attributions
        :param reg_lambda: Regularization parameter of the sparse linear surrogate model
        :return: A list of attributions of the (sentences) of the context
        """

        if self.response is None:
            self.generate_response()

        # uniformly sample n ablation vectors, compute logits using ablated context
        ablation_vectors = self.get_ablations(len(self.context_split), self.num_ablations)
        dataset = self.get_dataset(ablation_vectors)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding="longest")
        data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator)
        total_logits = ch.zeros((self.num_ablations, len(self.response_ids[0])), device=self.device)
        i = 0
        for batch in tqdm(data_loader):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            # print(len(batch['input_ids'][0]))
            with ch.no_grad():
                output = self.model(**batch)
            logits = output.logits[:, -(len(self.response_ids[0]) + 1): -1]
            labels = batch["labels"][:, -len(self.response_ids[0]):]
            batch_size, _ = labels.shape
            # print("LOGITS:")
            # print(self.response_ids.shape)
            # print(logits.shape)
            # print(logits.reshape(labels.shape[0] * labels.shape[1], -1).shape)
            # print(labels.shape)
            # print(labels.reshape(labels.shape[0] * labels.shape[1]).shape)
            # print()
            batch_size, seq_length = labels.shape
            # [num_tokens x vocab_size]
            reshaped_logits = logits.reshape(batch_size * seq_length, -1)
            reshaped_labels = labels.reshape(batch_size * seq_length)
            correct_logits = reshaped_logits.gather(-1, reshaped_labels[:, None])[:, 0]
            cloned_logits = reshaped_logits.clone()
            cloned_logits.scatter_(-1, reshaped_labels[:, None], -ch.inf)
            other_logits = cloned_logits.logsumexp(dim=-1)
            reshaped_outputs = correct_logits - other_logits
            total_logits[i:i + batch_size] = reshaped_outputs.reshape(batch_size, seq_length)
            i += self.batch_size
            # TODO: FIX
            # cur_logit_probs = _compute_logit_probs(logits, labels)
            # logit_probs[start_index: start_index + batch_size] = cur_logit_probs
            # start_index += batch_size
        # learn sparse linear surrogate model - weights will be the attributions
        log_probs = ch.nn.functional.logsigmoid(total_logits).sum(dim=1)
        log_1mprobs = ch.log1p(-ch.exp(log_probs))
        clf = Lasso(alpha=self.lasso_alpha)
        clf.fit(ablation_vectors, (log_probs - log_1mprobs).cpu() / len(self.response_ids[0]))
        # print("Lasso Coefficients:")
        # print(clf.coef_)
        self.attribution = clf.coef_
        return self.attribution * len(self.response_ids[0])


if __name__ == "__main__":
    # print(ch.cuda.is_available())
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    context = """
Attention Is All You Need

Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
1 Introduction
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht-1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
    """
    query = "What type of GPUs did the authors use in this paper?"

    attributor = Attributor.from_pretrained(model_name, context, query, device="cuda")
    print(attributor.get_response())
    start = time()
    result = attributor.get_attributions()
    end = time()
    indices = np.argsort(result)
    # check fitting, change creation
    for i in indices:
        print(result[i], attributor.context_split[i])
    print("Run time:", end-start)
    # TODO: implement top-k log drop and LDS - evaluator module
