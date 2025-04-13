from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as ch
import nltk
from itertools import product
from sklearn.linear_model import Lasso
import numpy as np


class Attributor:
    def __init__(self, model_name: str, context: str, query: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.context = context
        self.query = query
        self.context_split = nltk.sent_tokenize(context)
        # different datasets might need different templates
        self.prompt_template = "Context: {context}\n\nQuery: {query}"
        self.response = None
        self.response_ids = None
        self.attribution = None

    # generate all possible ablation vectors - only suitable for very small data
    def all_ablations(self, length):
        return np.array([list(p) for p in product([0, 1], repeat=length)])

    # return n random ablation vectors
    def get_ablations(self, length, n):
        return self.all_ablations(length)[np.random.choice(int(np.exp2(length) - 1), size=n, replace=False)]

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
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        with ch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=512)

        self.response_ids = output_ids[:, input_ids.shape[1]:]
        # self.full_text = self.tokenizer.decode(output_ids[0])
        self.response = self.tokenizer.decode(self.response_ids[0], add_special_tokens=False)

    def get_logit(self, prompt):
        with ch.no_grad():
            output = self.model(prompt)

        logits = output.logits
        log_probs = ch.nn.functional.log_softmax(logits, dim=-1)

        response_probs = []
        for i in range(self.response_ids.shape[1]):
            token_id = self.response_ids[0, i]
            token_log_prob = log_probs[0, prompt.shape[1] - self.response_ids.shape[1] + i - 1, token_id].item()
            response_probs.append(token_log_prob)
        sum_log_probs = np.sum(response_probs)

        # change back to logit
        return sum_log_probs - ch.log1p(-ch.exp(ch.tensor(sum_log_probs))).item()

    def get_attributions(self, num_ablations, reg_lambda):
        """
        Finds which parts of context attribute the most to the response generation

        :param num_ablations: Number of ablations to learn a sparse linear surrogate model of attributions
        :param reg_lambda: Regularization parameter of the sparse linear surrogate model
        :return: A list of attributions of the (sentences) of the context
        """

        if self.response is None:
            self.generate_response()

        # uniformly sample n ablation vectors, compute logits using ablated context
        ablation_vectors = self.get_ablations(len(self.context_split), num_ablations)
        print(ablation_vectors)
        logits = []
        for ablation in ablation_vectors:
            ablated_context = self.create_ablated_context(ablation)
            ablated_prompt = [
                {"role": "user", "content": self.prompt_template.format(context=ablated_context, query=self.query)}]
            ablated_prompt = self.tokenizer.apply_chat_template(ablated_prompt, tokenize=False,
                                                                add_generation_prompt=True) + self.response
            print("Ablated Prompt:")
            print(ablated_prompt)
            ablated_ids = self.tokenizer.encode(ablated_prompt, return_tensors="pt")
            logits.append(self.get_logit(ablated_ids))
            print("Logit:", logits[-1])
            print("Prob from logit:", ch.sigmoid(ch.tensor(logits[-1])).item())
            print()

        # learn sparse linear surrogate model - weights will be the attributions
        clf = Lasso(alpha=reg_lambda)
        clf.fit(ablation_vectors, logits)
        print("Lasso Coeficcients:")
        print(clf.coef_)
        self.attribution = clf.coef_
        return self.attribution


if __name__ == "__main__":
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

    attributor = Attributor(model_name, context, query)
    result = attributor.get_attributions(100, 0.1)
    index = np.argmax(result)
    print(print(result[index], attributor.context_split[index]))
