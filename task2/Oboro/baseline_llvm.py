import argparse
import json
import time
from typing import List, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: AutoTokenizer,
    trust_remote_code: bool,
) -> Tuple[float, List[int], List[int]]:

    llm = LLM(
        model=model,
        tensor_parallel_size=2,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=0.95,
        max_model_len=65536,
        dtype=torch.bfloat16
    )

    input_num_tokens = []
    output_num_tokens = []
    start = time.perf_counter()
    
    for prompt, _, output_len in tqdm(requests):
        sampling_params = SamplingParams(
            n=1,
            best_of=1,
            presence_penalty=1.0,
            frequency_penalty=0.0,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            use_beam_search=False,
            length_penalty=1,
            early_stopping=False,
            ignore_eos=False,
            max_tokens=output_len,
            logprobs=None,
            prompt_logprobs=None,
            skip_special_tokens=True,
        )
        llm_outputs = llm.generate(prompt, sampling_params)
        token_ids = llm_outputs[0].outputs[0].token_ids
        decoded_tokens = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        input_num_tokens.append(len(tokenizer(prompt, return_tensors="pt").input_ids[0]))
        output_num_tokens.append(len(token_ids))

    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens

def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = load_requests(args)

    elapsed_time, input_num_tokens, output_num_tokens = run_hf(requests, args.model, tokenizer, args.trust_remote_code)
    prompt_num_tokens = sum(input_num_tokens)
    total_num_tokens = sum(output_num_tokens)

    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s")
    print(f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s")
    print(f"Prompt_num_tokens: {prompt_num_tokens:.2f} tokens")
    print(f"Total_num_tokens: {total_num_tokens:.2f} tokens")

def load_requests(args: argparse.Namespace) -> List[Tuple[str, int, int]]:
    if args.dataset is None:
        prompt = "hi" * (args.input_len - 1)
        return [(prompt, args.input_len, args.output_len) for _ in range(args.num_samples)]
    else:
        with open(args.dataset) as f:
            return json.load(f)[:args.num_samples]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--model", type=str, required=True, help="Model identifier.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer identifier.")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request.")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples used for inference test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument('--trust-remote-code', action='store_true', help='Trust remote code from HuggingFace.')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], help='Data type for model weights and activations.')

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    assert (args.dataset is None) != (args.input_len is None or args.output_len is None), "Specify either --dataset or both --input-len and --output-len."

    main(args)
