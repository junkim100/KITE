from openai import OpenAI
from dotenv import load_dotenv
import json
import os
from threading import Semaphore, Thread
import time
from fire import Fire

api_semaphore = Semaphore(value=5)

# Load the API key from the .env file
load_dotenv()
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def label_relevance(text):
    text_lower = text.lower()
    if 'irrelevant' in text_lower or 'not relevant' in text_lower:
        return 'irrelevant'
    elif 'relevant' in text_lower:
        return 'relevant'
    else:
        return text_lower[:10]

def filter_data(model, shot, text):
    user_input = {
        "role": "user",
        "content": text
    }

    shot.append(user_input)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=shot
        )

        response_text = response.choices[0].message.content
        return label_relevance(response_text)
    except Exception as e:
        print(f"Error: {e}")
        return None

def filter_MMLU(input_folder, output_folder, filter_shot, translate_text, model='gpt-4-turbo-2024-04-09', debug=False, verbose=False):
    if verbose: print("\n//////////////////////////////////////////////Filtering MMLU data//////////////////////////////////////////////\n")
    if not os.path.exists(input_folder): os.makedirs(input_folder)
    # Processing each item in the 'rows' array
    with open(os.path.join(input_folder, 'MMLU_filtered.json'), 'w') as positive_file, \
        open(os.path.join(input_folder, 'MMLU_removed.json'), 'w') as negative_file:
        for filename in os.listdir(output_folder):
            if filename.endswith('.json') and filename not in ['MMLU_filtered.json','MMLU_removed.json']:
                # Construct the full path to the file
                file_path = os.path.join(output_folder, filename)
                # Process each file
                with open(file_path, 'r') as json_file:
                    for line_number, line in enumerate(json_file):
                        if debug and line_number==3: break
                        if verbose: print(f"Processing line {line_number}")
                        try:
                            data = json.loads(line.strip())
                            prompt_text = data['question']
                            relevancy = filter_data(model, filter_shot, f"question: {prompt_text}\nchoices: {data['choices']}\nanswer: {data['answer']}")
                            translated = translate_text(prompt_text)

                            english_dict = {
                                'question': prompt_text,
                                'choices': data['choices'],
                                'answer': data['answer']
                            }
                            korean_dict = {
                                'question': translated,
                                'choices': [translate_text(a) if any(char.isalpha() for char in a) else a for a in data['choices']],
                                'answer': translate_text(str(data['answer'])) if any(char.isalpha() for char in str(data['answer'])) else str(data['answer'])
                            }

                            if relevancy == 'relevant':
                                positive_file.write(json.dumps({
                                                                'english': english_dict,
                                                                'korean': korean_dict
                                                                },
                                                            ensure_ascii=False) + '\n')
                            elif relevancy == 'irrelevant':
                                negative_file.write(json.dumps({
                                                                'english': english_dict,
                                                                'korean': korean_dict
                                                                },
                                                            ensure_ascii=False) + '\n')
                            else:
                                if verbose: print(f"Invalid relevancy: {relevancy} for: {prompt_text}")
                        except json.JSONDecodeError:
                            if verbose: print(f"Skipping invalid JSON at line {line_number}: {line}")

if __name__ == '__main__':
    Fire(filter_MMLU)