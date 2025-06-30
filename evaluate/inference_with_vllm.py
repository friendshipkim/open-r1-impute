#!/usr/bin/env python3
"""
Script to load Hugging Face dataset, select subset as prompts, 
run inference using VLLM with Qwen2.5 model, and save results.

Usage:
python inference_with_vllm.py \
    --model-path "./open-r1-impute/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-oracle-lr2e-6-warmup0.05" \
    --output-file "./temp.jsonl" \
    --dataset-name "HuggingFaceH4/ultrachat_200k" \
    --split "test_sft" \
    --prompt-column "prompt" \
    --subset-size 100 \
    --seed 42 \
    --include-original-data
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


def load_and_sample_dataset(dataset_name: str, split: str = "train", 
                           subset_size: int = None, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load Hugging Face dataset and sample a subset.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split to use (train, test, validation)
        subset_size: Number of examples to sample (None for all)
        seed: Random seed for reproducibility
    
    Returns:
        List of dataset examples
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    if subset_size and subset_size < len(dataset):
        print(f"Sampling {subset_size} examples from {len(dataset)} total examples")
        dataset = dataset.shuffle(seed=seed).select(range(subset_size))
    else:
        print(f"Using all {len(dataset)} examples")
    
    return dataset


def format_prompts(dataset: List[Dict[str, Any]], prompt_column: str, 
                  prompt_template: str = None) -> List[str]:
    """
    Format dataset examples into prompts for inference.
    
    Args:
        dataset: List of dataset examples
        prompt_column: Column name containing the prompt text
        prompt_template: Template string for formatting prompts
    
    Returns:
        List of formatted prompts
    """
    prompts = []
    
    for example in dataset:
        if prompt_column not in example:
            raise ValueError(f"Column '{prompt_column}' not found in dataset. Available columns: {list(example.keys())}")
        
        prompt_text = example[prompt_column]
        
        if prompt_template:
            # Use template if provided
            formatted_prompt = prompt_template.format(prompt=prompt_text)
        else:
            # Use raw prompt text
            formatted_prompt = prompt_text
        
        prompts.append(formatted_prompt)
    
    return prompts


def run_inference(model_path: str, prompts: List[str], 
                 temperature: float = 0.7, top_p: float = 0.95,
                 max_tokens: int = 512, batch_size: int = 8) -> List[Dict[str, Any]]:
    """
    Run inference using VLLM.
    
    Args:
        model_path: Path to the model
        prompts: List of prompts to generate responses for
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
    
    Returns:
        List of generation results
    """
    print(f"Loading model from: {model_path}")
    
    # Initialize VLLM model
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    print(f"Running inference on {len(prompts)} prompts...")
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Generate responses
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process outputs
        for j, output in enumerate(outputs):
            result = {
                "prompt": batch_prompts[j],
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "usage": {
                    "prompt_tokens": output.prompt_logprobs,
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                }
            }
            results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_file: str, 
                include_original_data: bool = False, original_dataset: List[Dict[str, Any]] = None):
    """
    Save inference results to file.
    
    Args:
        results: List of inference results
        output_file: Path to output file
        include_original_data: Whether to include original dataset fields
        original_dataset: Original dataset examples
    """
    print(f"Saving results to: {output_file}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            if include_original_data and original_dataset and i < len(original_dataset):
                # Combine original data with generation results
                combined_result = {**original_dataset[i], **result}
            else:
                combined_result = result
            
            f.write(json.dumps(combined_result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on Hugging Face dataset using VLLM with Qwen2.5 model"
    )
    
    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceH4/ultrachat_200k",
                       help="Name of the Hugging Face dataset")
    parser.add_argument("--split", type=str, default="test_sft",
                       help="Dataset split to use (default: test_sft)")
    parser.add_argument("--prompt-column", type=str, default="prompt",
                       help="Column name containing the prompt text")
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Number of examples to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling (default: 42)")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the model directory")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95,
                       help="Top-p sampling parameter (default: 0.95)")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference (default: 8)")
    
    # Prompt formatting
    parser.add_argument("--prompt-template", type=str, default=None,
                       help="Template string for formatting prompts (use {prompt} placeholder)")
    
    # Output arguments
    parser.add_argument("--output-file", type=str, required=True,
                       help="Path to output file (JSONL format)")
    parser.add_argument("--include-original-data", action="store_true",
                       help="Include original dataset fields in output")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # # load huggingface model and upload to hub
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model_name = args.model_path.split("/")[-1]
    # model = AutoModelForCausalLM.from_pretrained(args.model_path).to(torch.bfloat16)
    # model.config.torch_dtype = torch.bfloat16
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # model.push_to_hub(f"friendshipkim/{model_name}")
    # tokenizer.push_to_hub(f"friendshipkim/{model_name}")
    # print(f"Uploaded model to friendshipkim/{model_name}")
    # exit()
    
    # Load and sample dataset
    dataset = load_and_sample_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        subset_size=args.subset_size,
        seed=args.seed
    )
    
    # Format prompts
    prompts = format_prompts(
        dataset=dataset,
        prompt_column=args.prompt_column,
        prompt_template=args.prompt_template
    )
    
    # Run inference
    results = run_inference(
        model_path=args.model_path,
        prompts=prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size
    )
    
    # Save results
    save_results(
        results=results,
        output_file=args.output_file,
        include_original_data=args.include_original_data,
        original_dataset=dataset if args.include_original_data else None
    )
    
    print("Inference completed successfully!")



if __name__ == "__main__":
    main() 