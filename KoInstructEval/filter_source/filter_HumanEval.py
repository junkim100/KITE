from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import re
from threading import Semaphore, Thread
import time
from fire import Fire

api_semaphore = Semaphore(value=5)

# Load the API key from the .env file
load_dotenv()
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def replace_triple_quoted_text_with_translation(model, translate_text, text):
    # Pattern to match text enclosed in triple quotes
    pattern = r'("""(.*?)"""|\'\'\'(.*?)\'\'\')'

    # Function to replace matched text with its translation
    def replace_with_translation(match):
        # Extract the matched text
        matched_text = match.group(2) if match.group(2) else match.group(3)

        try:
            translated_text = translate_text(matched_text)
        except Exception as e:
            print("Error during translation: ", e)  # Capture and print any error during translation
            return match.group(0)  # Return the original text in case of error

        return match.group(1)[0:3] + translated_text + match.group(1)[0:3]

    # Replace all occurrences of the pattern with their translations
    translated_text = re.sub(pattern, replace_with_translation, text, flags=re.DOTALL)
    return translated_text

def filter_HumanEval(input_folder, output_folder, translate_text, model='gpt-4-turbo-2024-04-09', debug=False, verbose=False):
    if verbose: print("\n//////////////////////////////////////////////Filtering HumanEval data//////////////////////////////////////////////\n")
    if not os.path.exists(input_folder): os.makedirs(input_folder)
    # Processing each item in the 'rows' array
    with open(os.path.join(input_folder, 'HumanEval_filtered.json'), 'w') as translated_file:
        with open(os.path.join(output_folder, 'HumanEval.json'), 'r') as json_file:
            for line_number, line in enumerate(json_file):
                if debug and line_number==3: break
                if verbose: print(f"Processing line {line_number}")
                try:
                    data = json.loads(line.strip())
                    prompt_text = data['prompt']
                    translated = replace_triple_quoted_text_with_translation(model, translate_text, prompt_text)

                    input_dict = {
                        'question': prompt_text
                    }
                    korean_dict = {
                        'question': translated
                    }

                    output_data = {
                        'input': input_dict,
                        'korean': korean_dict,
                        'answer': data['canonical_solution']
                    }

                    translated_file.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    if verbose: print(f"Skipping invalid JSON at line {line_number}: {line}")

if __name__ == '__main__':
    Fire(filter_HumanEval)