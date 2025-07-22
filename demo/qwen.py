import sys
sys.path.append("../")
from genesis.models.qwen import QwenForCausalLM, ModelArgs
import genesis
import torch
import random
from safetensors.torch import load_file
from transformers import AutoTokenizer

MODEL_PATH = "/root/workspace/models_hub/Qwen2.5-0.5B-Instruct/"

if __name__ == "__main__":
    config = ModelArgs()
    model = QwenForCausalLM(config)
    state_dict = load_file(MODEL_PATH + "model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    q = "Where is the capital of China?"
    message = "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
    instruct = tokenizer.encode(message)
    # print(instruct)
    a = genesis.Tensor([instruct]).to(genesis.device("cuda")).long()

    print(message, end="", flush=True)

    def generate(model, input_tensor, max_length=100, temperature=0.5, top_k=5):
        model.eval()  # Set model to evaluation mode
        generated_tokens = list(map(int, list(input_tensor.detach().cpu().numpy()[0])))

        for _ in range(max_length):
            logits = model(input_tensor)
            logits = logits[:, -1, :] / temperature
            probabilities = genesis.nn.Softmax(dim=-1)(logits)
            top_k_values, top_k_indices = torch.topk(probabilities.data.data, k=top_k, dim=-1)
            top_k_indices = top_k_indices.tolist()[0]
            top_k_values = top_k_values.tolist()[0]
            #print("Top 5 indices:", top_k_indices[:5], ", ".join([tokenizer.decode([idx]) for idx in top_k_indices[0:5]]))
            #print("Top 5 probabilities:", top_k_values[:5])

            #next_token = torch.argmax(probabilities.data.data, dim=-1).item()
            top_k_probabilities = [prob / sum(top_k_values) for prob in top_k_values]
            next_token = random.choices(top_k_indices, weights=top_k_probabilities, k=1)[0]

            #print(next_token)
            generated_tokens.append(next_token)

            #print(tokenizer.decode(generated_tokens), end='', flush=True)
            print(tokenizer.decode(next_token), end='', flush=True)

            if next_token == 151645:
                break
            input_tensor = genesis.Tensor([generated_tokens]).to(genesis.device("cuda")).long()

        #print(message + tokenizer.decode(generated_tokens))
        return generated_tokens

    generate(model, a, max_length=100)