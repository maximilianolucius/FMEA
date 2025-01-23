"""
Unified evaluation pipeline for language models across diverse benchmarks.

This script handles end-to-end evaluation workflows including:
- Inference generation with vLLM or OpenAI models
- Correctness checking of model responses
- Token usage tracking and result management
- Support for multiple datasets and task formats

Key features:
- Parallel execution of inference and evaluation
- Temperature-controlled sampling
- Resume capabilities for interrupted evaluations
- Difficulty filtering for math problems
- Multi-sample generation support

Architecture:
- TaskHandler subclasses handle dataset-specific logic
- Separate workflows for inference generation (perform_inference_and_save) 
  and correctness checking (perform_check)
- Integrated token cost tracking for OpenAI and vLLM backends
"""

import json
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from tqdm import tqdm
from util.task_handlers import *
from util.model_utils import *
from openai import OpenAI
import concurrent.futures
from functools import partial


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling numpy arrays in results."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def fetch_response_openai(llm, model_name, max_tokens, temp, prompt):
    """
    Execute OpenAI API call with model-specific handling.

    Args:
        llm: OpenAI client instance
        model_name: Model identifier (e.g., "gpt-4")
        max_tokens: Maximum tokens to generate
        temp: Sampling temperature
        prompt: Conversation-style prompt format

    Returns:
        OpenAI response object with special handling for O1 models
    """
    model_name = model_name.replace("openai/", "")
    if "o1" in model_name:
        # O1 models require user role conversion and fixed temperature
        for p in prompt:
            p["role"] = "user"

        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=1,
            temperature=1,  # O1 requires fixed temperature
            max_completion_tokens=max_tokens
        )
    else:
        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=1,
            temperature=temp,
            max_tokens=max_tokens
        )
    return response


def perform_inference_and_check(handler: TaskHandler, temperatures, max_tokens, result_file, llm, system_prompt, args):
    """
    Execute full inference-checking pipeline with parallel processing.

    Steps:
    1. Load existing results and filter unprocessed items
    2. Generate model responses for remaining items
    3. Validate responses using parallel workers
    4. Aggregate and save results with token usage statistics

    Args:
        handler: Dataset-specific TaskHandler instance
        temperatures: List of sampling temperatures
        max_tokens: Maximum generation length
        result_file: Path to save results JSON
        llm: Initialized model client (vLLM or OpenAI)
        system_prompt: Model-specific instruction template
        args: Command line arguments
    """
    results = handler.load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")
    train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
                                                 filter_difficulty=args.filter_difficulty, args=args)
    remaining_data = handler.process_remaining_data(train_data, results)
    conversations = handler.make_conversations(remaining_data, system_prompt, args.model)

    for temp in temperatures:
        # Generate responses with appropriate backend
        if args.model.startswith("openai"):
            fetch_partial = partial(fetch_response_openai, llm, args.model, max_tokens, temp)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
                responses = list(e.map(fetch_partial, conversations))
        else:
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temp)
            responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)

        # Process responses with parallel validation
        total_correct = 0
        total_finish = 0
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_task = {}
            token_usages = {}
            for idx, response in enumerate(responses):
                if args.model.startswith("openai"):
                    response_str = response.choices[0].message.content.strip()
                    token_usages[idx] = response.usage
                else:
                    response_str = response.outputs[0].text.strip()
                    token_usages[idx] = {
                        "completion_tokens": len(response.outputs[0].token_ids),
                        "prompt_tokens": len(response.prompt_token_ids)
                    }
                future_to_task[executor.submit(handler.update_results, remaining_data[idx], response_str)] = idx

            # Aggregate validation results
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing Generations"):
                idx = future_to_task[future]
                response_entry = future.result()
                total_correct += response_entry["correctness"]
                total_finish += 1
                problem_key = remaining_data[idx][handler.get_question_key()]

                # Update results structure
                if problem_key not in results:
                    results[problem_key] = {
                        **remaining_data[idx],
                        "responses": {},
                        "token_usages": {},
                        "prompt": conversations[idx][1]["content"]
                    }
                results[problem_key]["responses"][str(temp)] = response_entry
                results[problem_key]["token_usages"][str(temp)] = token_usages[idx]

        # Print temperature-specific accuracy
        acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
        print(json.dumps({"acc": acc}))

    # Save token usage statistics
    token_usage_dir = os.path.join(os.path.dirname(result_file), "token_usage")
    os.makedirs(token_usage_dir, exist_ok=True)
    token_usage_result_file = os.path.join(token_usage_dir, os.path.basename(result_file))

    with open(token_usage_result_file, "w") as f:
        json.dump({
            "completion_tokens": sum(
                u.get("completion_tokens", 0) for r in results.values() for u in r["token_usages"].values()),
            "prompt_tokens": sum(
                u.get("prompt_tokens", 0) for r in results.values() for u in r["token_usages"].values())
        }, f, indent=4)

    # Save final results
    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def perform_check(handler: TaskHandler, temperatures, result_file, args):
    """
    Re-evaluate existing responses for correctness.

    Used to update evaluation metrics without re-running inference.

    Args:
        handler: TaskHandler for dataset-specific evaluation
        temperatures: List of temperatures to recheck
        result_file: Path to existing results JSON
        args: Command line arguments
    """
    results = handler.load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")

    train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
                                                 filter_difficulty=args.filter_difficulty, args=args)
    remaining_data = handler.process_remaining_data(train_data, {})

    # Identify responses needing revalidation
    tasks = []
    for item in remaining_data:
        problem_key = item[handler.get_question_key()]
        if problem_key in results and "responses" in results[problem_key]:
            for temp in temperatures:
                if str(temp) in results[problem_key]["responses"]:
                    for sample_id, response_entry in enumerate(results[problem_key]["responses"][str(temp)]):
                        if sample_id > (args.n - 1): continue
                        tasks.append((item, temp, response_entry["content"], sample_id))

    print(f"Revalidating {len(tasks)} responses...")

    # Parallel revalidation
    total_correct = 0
    correct = {temp: {} for temp in temperatures}
    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {
            executor.submit(handler.update_results, item, content): (item, temp, sample_id)
            for (item, temp, content, sample_id) in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Revalidation"):
            item, temp, sample_id = future_to_task[future]
            new_entry = future.result()
            total_correct += new_entry["correctness"]

            problem_key = item[handler.get_question_key()]
            correct[temp][problem_key] = new_entry["correctness"]
            results[problem_key]["responses"][str(temp)][sample_id] = new_entry

    # Print temperature-wise accuracy
    for temp in temperatures:
        temp_correct = sum(correct[temp].values())
        temp_acc = round(temp_correct / len(correct[temp]), 4) if correct[temp] else 0
        print(f"Temperature {temp} acc: {temp_correct}/{len(correct[temp])} ({temp_acc})")

    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def perform_inference_and_save(handler: TaskHandler, temperatures, max_tokens, result_file, llm, system_prompt, args):
    """
    Batch inference generation without immediate validation.

    Args:
        handler: TaskHandler for dataset processing
        temperatures: List of sampling temperatures
        max_tokens: Generation length limit
        result_file: Output JSON path
        llm: Initialized model client
        system_prompt: Instruction template
        args: Command line arguments
    """
    results = handler.load_existing_results(result_file)
    train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
                                                 filter_difficulty=args.filter_difficulty, args=args)
    remaining_data = handler.process_remaining_data(train_data, results)
    conversations = handler.make_conversations(remaining_data, system_prompt, args.model)

    # Generate responses for all temperatures
    for temp in temperatures:
        if args.model.startswith("openai"):
            fetch_partial = partial(fetch_response_openai, llm, args.model, max_tokens, temp)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
                responses = list(e.map(fetch_partial, conversations))
        else:
            sampling_params = SamplingParams(n=args.n, max_tokens=max_tokens, temperature=temp)
            responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)

        # Store responses and token usage
        for idx, response in enumerate(responses):
            problem_key = remaining_data[idx][handler.get_question_key()]
            if problem_key not in results:
                results[problem_key] = {
                    **remaining_data[idx],
                    "responses": {},
                    "token_usages": {},
                    "prompt": conversations[idx][1]["content"]
                }

            # Extract multiple samples
            samples = []
            token_usages = []
            for sample_idx in range(args.n):
                content = response.choices[0].message.content.strip() if args.model.startswith("openai") else \
                response.outputs[sample_idx].text.strip()
                samples.append({
                    "content": content,
                    "correctness": None,
                    "reason": None
                })
                if not args.model.startswith("openai"):
                    token_usages.append({
                        "completion_tokens": len(response.outputs[sample_idx].token_ids),
                        "prompt_tokens": len(response.prompt_token_ids)
                    })

            results[problem_key]["responses"][str(temp)] = samples
            results[problem_key]["token_usages"][str(temp)] = token_usages if token_usages else response.usage

    # Save final outputs
    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def main():
    """
    Main execution flow for unified evaluator.

    Handles:
    - Argument parsing
    - Task handler initialization
    - Backend selection (vLLM/OpenAI)
    - Routing to inference/checking workflows
    """
    parser = argparse.ArgumentParser(description="Unified evaluation pipeline for language models.")
    parser.add_argument("--dataset", required=True, choices=TASK_HANDLERS.keys(), help="Evaluation dataset name")
    parser.add_argument("--model", required=True, help="Model identifier or path")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism degree")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max generation tokens")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate")
    parser.add_argument("--source", help="Data source filter")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index")
    parser.add_argument("--filter-difficulty", action="store_true", help="Enable difficulty filtering")
    parser.add_argument("--result-dir", default="./", help="Output directory")
    parser.add_argument("--check", action="store_true", help="Run evaluation checks only")
    parser.add_argument("--inference", action="store_true", help="Run inference only")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0], help="Sampling temperatures")
    parser.add_argument("--math-difficulty-lower-bound", type=int, help="Minimum math difficulty")
    parser.add_argument("--math-difficulty-upper-bound", type=int, help="Maximum math difficulty")
    parser.add_argument("--n", type=int, default=1, help="Samples per prompt")
    args = parser.parse_args()

    # Initialize dataset handler
    handler = TASK_HANDLERS[args.dataset]()

    # Temperature handling for special models
    temperatures = [1] if args.model.startswith("openai/o1") else args.temperatures

    # Result file naming
    file_suffix = f"{args.math_difficulty_lower_bound}_{args.math_difficulty_upper_bound}" \
        if args.math_difficulty_lower_bound or args.math_difficulty_upper_bound else ""
    result_file = os.path.join(args.result_dir,
                               f"{MODEL_TO_NAME[args.model]}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}{file_suffix}.json")

    # Route to appropriate workflow
    if args.check:
        perform_check(handler, temperatures, result_file, args)
    else:
        llm = OpenAI() if args.model.startswith("openai") else LLM(model=args.model, tensor_parallel_size=args.tp)
        system_prompt = SYSTEM_PROMPT[args.model]
        if args.inference:
            perform_inference_and_save(handler, temperatures, args.max_tokens, result_file, llm, system_prompt, args)
        else:
            perform_inference_and_check(handler, temperatures, args.max_tokens, result_file, llm, system_prompt, args)


if __name__ == "__main__":
    main()