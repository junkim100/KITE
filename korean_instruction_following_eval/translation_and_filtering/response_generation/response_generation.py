import logging
import fire
from . import models


def generate_response(
    model_type: str,
    model_name: str,
    input_file: str,
    output_file: str,
    shot_num: int,
    verbosity: int = 0,
):
    """
    Run the specified model on the input file and save the results to the output file.

    Args:
        model_type (str): The type of model ('hf' for HuggingFace, 'openai' for OpenAI).
        model_name (str): The name of the model to use (e.g., 'hyperCLOVA X', 'SOLAR', or a HuggingFace/OpenAI model name).
        input_file (str): The path to the input file in JSONL format.
        output_file (str): The path to the output file to save responses in JSONL format.
        shot_num (int): The number of shots to use for the OpenAI model (default: 0).s
        verbosity (int): The verbosity level for logging (-1 for silent, 0 for INFO, 1 for DEBUG).
    """
    if verbosity == -1:
        logging.basicConfig(level=logging.CRITICAL)
    elif verbosity == 0:
        logging.basicConfig(level=logging.INFO)
    elif verbosity == 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        raise ValueError(
            "Invalid verbosity level. Choose -1 for silent, 0 for INFO, 1 for DEBUG."
        )

    if model_type == "hf":
        model = models.HuggingFaceModel(model_name)
    elif model_type == "openai":
        model = models.OpenAIModel(model_name)
    elif model_type == "clova":
        model = models.HyperCLOVA_X(model_name)
    elif model_type == "solar":
        model = models.SolarAPI(model_name)
    else:
        raise ValueError(
            "Invalid model_type. Choose 'hf' for HuggingFace, 'openai' for OpenAI, 'clova' for HyperCLOVA X, 'solar' for SOLAR-API"
        )

    model.process_prompts(input_file, output_file, shot_num)


if __name__ == "__main__":
    fire.Fire({"run": generate_response})
