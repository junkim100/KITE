import json


def load_jsonl(file_path):
    """Load a JSONL file and return a list of JSON objects."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(data, file_path):
    """Save a list of JSON objects to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    # Load the data from the JSONL file
    data = load_jsonl("culturally_aware.jsonl")

    # Process each item to retain only the 'kwargs' field
    processed_data = [{"kwargs": item["kwargs"]} for item in data if "kwargs" in item]

    # Save the processed data to a new JSONL file
    save_jsonl(processed_data, "culturally_aware.jsonl")

    print("The file has been processed and saved with only 'kwargs' fields.")


if __name__ == "__main__":
    main()
