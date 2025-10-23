import os
import subprocess
import fire
import logging
from translation_and_filtering.response_generation.response_generation import (
    generate_response,
)


def response_generation(
    model, instruction_file, response_output_dir, shot_num, verbosity
):
    output_file = os.path.join(
        response_output_dir, f"{model['name'].replace('/', '_')}.jsonl"
    )
    generate_response(
        model_type=model["type"],
        model_name=model["name"],
        input_file=instruction_file,
        output_file=output_file,
        shot_num=shot_num,
        verbosity=verbosity,
    )


def run_eval(model, instruction_file, response_output_dir, eval_output_dir, verbosity):
    model_name = model["name"].replace("/", "_")
    input_response_file = os.path.join(response_output_dir, f"{model_name}.jsonl")
    output_file = os.path.join(eval_output_dir, model_name)
    log_file = os.path.join(eval_output_dir, f"{model_name}_eval.txt")

    cmd = [
        "python",
        "-m",
        "korean_instruction_following_eval.eval.evaluation_main",
        "--input_data",
        instruction_file,
        "--input_response_data",
        input_response_file,
        "--output_file",
        output_file,
        "--verbosity",
        str(verbosity),
    ]

    with open(log_file, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=log)

    logging.info(f"Saved evaluation results to {log_file}")


def main(
    model_type,
    model,
    instruction_file,
    response_output_dir,
    eval_output_dir,
    shot_num=0,
    verbosity=0,
):
    logging_levels = {1: logging.DEBUG, 0: logging.INFO, -1: logging.WARNING}
    logging.basicConfig(level=logging_levels.get(verbosity, logging.INFO))

    model_dict = {"type": model_type, "name": model}

    logging.info("Starting response generation...")
    response_generation(
        model_dict, instruction_file, response_output_dir, shot_num, verbosity
    )
    logging.info("Response generation complete.")

    logging.info("Starting evaluation...")
    run_eval(
        model_dict, instruction_file, response_output_dir, eval_output_dir, verbosity
    )
    logging.info("Evaluation complete.")


if __name__ == "__main__":
    fire.Fire(main)
