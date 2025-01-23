"""
Automate running multiple evaluations on a language model with different configurations.

This script executes specified benchmark evaluations on a model by calling an external inference script,
captures accuracy metrics from the output logs, and writes consolidated results to a file.

Key components:
- Maps evaluation names to dataset splits (eval_to_split)
- Accepts CLI arguments for model path, evaluations list, and parameters
- Runs subprocesses for each evaluation using unified_evaluator.py
- Parses accuracy metrics from JSON-formatted outputs
- Handles log collection and error reporting

Dependencies:
- Requires unified_evaluator.py in the same directory for actual inference execution
"""

import argparse
import subprocess
import os
import json

# Mapping of evaluation names to their respective dataset splits
eval_to_split = {
    "MATH500": "test",
    "AIME": "train",
    "GPQADiamond": "train",
    "MMLU": "test",
    "MMLUPro": "test",
    "LiveCodeBench": "test",
    "GSM8K": "test",
    "ARC-C": "test",
}


def parse_arguments():
    """Parse command-line arguments for model evaluation configuration.

    Returns:
        argparse.Namespace: Object containing parsed arguments with attributes:
        - model (str): Path to model file/directory
        - evals (str): Comma-separated list of evaluations to run
        - tp (int): Tensor parallelism degree (default: 8)
        - filter_difficulty (bool): Flag to enable difficulty filtering
        - source (str): Data source for difficulty filtering
        - output_file (str): Path to output log file
        - temperatures (list[float]): Sampling temperatures to use
    """
    parser = argparse.ArgumentParser(description="Process model path, prompt format, and evals to run.")
    parser.add_argument("--model", required=True, type=str, help="Path to the model.")
    parser.add_argument("--evals", required=True, type=str, help="Comma-separated list of evals to run (no spaces).")
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
    parser.add_argument("--filter-difficulty", action="store_true", help="Filter difficulty.")
    parser.add_argument("--source", type=str, help="Source for the dataset.")
    parser.add_argument("--output_file", required=True, type=str, help="Output file to write results to.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0], help="Temperature for sampling.")
    return parser.parse_args()


def extract_accuracy_from_output(output):
    """Extract accuracy value from subprocess output by checking for JSON-formatted lines.

    Args:
        output (str): Combined stdout/stderr output from evaluation subprocess

    Returns:
        float/None: Last found accuracy value or None if not found
    """
    lines = output.splitlines()[::-1]
    for line in lines:
        try:
            data = json.loads(line.replace("'", '"'))
            if "acc" in data:
                return data["acc"]
        except json.JSONDecodeError:
            continue
    return None


def write_logs_to_file(logs, output_file):
    """Write collected logs to specified output file.

    Args:
        logs (str): Combined log content to write
        output_file (str): Path to target output file

    Prints success message or error if writing fails
    """
    try:
        with open(output_file, "w") as file:
            file.write(logs)
        print(f"Logs successfully written to {output_file}")
    except IOError as e:
        print(f"Failed to write logs to file {output_file}: {e}")


def main():
    """Main execution flow:
    1. Parse command line arguments
    2. Run specified evaluations through subprocesses
    3. Collect and parse accuracy results
    4. Write consolidated logs to output file
    5. Print final results summary
    """
    args = parse_arguments()
    model_path = args.model
    evals = args.evals.split(",")
    output_file = args.output_file
    tp = args.tp
    temperatures = [str(t) for t in args.temperatures]

    script_path = "unified_evaluator.py"
    all_logs = ""
    results = {}

    for eval_name in evals:
        command = [
            "python", script_path,
            "--model", model_path,
            "--dataset", eval_name,
            "--split", eval_to_split[eval_name],
            "--tp", str(tp),
            "--temperatures"
        ]
        command.extend(temperatures)

        if args.filter_difficulty:
            assert args.source != "", "No source passed for filtering difficulty."
            command += ["--filter-difficulty", "--source", args.source]

        all_logs += f"\nRunning eval: {eval_name} with command {command}\n"
        try:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                output_lines = []
                for line in proc.stdout:
                    print(line, end="")
                    output_lines.append(line)
                    all_logs += line
                proc.wait()

                if proc.returncode == 0:
                    output = "".join(output_lines)
                    results[eval_name] = extract_accuracy_from_output(output)
                else:
                    raise subprocess.CalledProcessError(proc.returncode, command)

        except subprocess.CalledProcessError as e:
            error_message = f"Error running {eval_name}: {e}\n"
            print(error_message)
            all_logs += error_message

    write_logs_to_file(all_logs, output_file)
    print("Results:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()