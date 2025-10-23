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

def filter_DROP(input_folder, output_folder, filter_shot, translate_text, model='gpt-4-turbo-2024-04-09', debug=False, verbose=False):
    if verbose: print("\n//////////////////////////////////////////////Filtering DROP data//////////////////////////////////////////////\n")
    if not os.path.exists(input_folder): os.makedirs(input_folder)
    # Processing each item in the 'rows' array
    with open(os.path.join(input_folder, 'DROP_filtered.json'), 'w') as positive_file, \
        open(os.path.join(input_folder, 'DROP_removed.json'), 'w') as negative_file:
        for filename in os.listdir(output_folder):
            if filename.endswith('.json') and filename not in ['DROP_filtered.json','DROP_removed.json']:
                # Construct the full path to the file
                file_path = os.path.join(output_folder, filename)
                # Process each file
                with open(file_path, 'r') as json_file:
                    passsage_relevancy = None
                    passage_text = None
                    count = 0

                    # Many data points have the same passage, so we can save time by only filtering the passage once
                    for l in json_file:
                        count += 1
                        if debug and count==2: break
                        data = json.loads(l)
                        temp_passsage_text = data['passage']

                        if passage_text!=temp_passsage_text:
                            passage_text = temp_passsage_text
                            passsage_relevancy = filter_data(model, filter_shot, passage_text)
                            passage_translated = translate_text(passage_text)
                        elif passsage_relevancy == 'irrelevant':
                            if verbose: print(f"\n\nSkipping irrelevant passage: {passage_text}\n\n")
                            continue

                        for line_number, line in enumerate(json_file):
                            data = json.loads(line) 
                            if debug and line_number==3: break
                            if verbose: print(f"Processing file {filename} line {line_number}")
                            try:
                                prompt_text = data['question']
                                relevancy = filter_data(model, filter_shot, f"question: {prompt_text}\nanswer: {data['answers_spans']['spans'][0]}")
                                translated = translate_text(prompt_text)

                                english_dict = {
                                    'passage': passage_text,
                                    'question': prompt_text,
                                    'answer': data['answers_spans']['spans'][0]
                                }
                                korean_dict = {
                                    'passage': passage_translated,
                                    'question': translated,
                                    'answer': translate_data(data['answers_spans']['spans'][0]) if any(char.isalpha() for char in data['answers_spans']['spans'][0]) else data['answers_spans']['spans'][0]
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
                            except KeyError:
                                if verbose: print(f"Skipping invalid Key at line {line_number}: {line}")

if __name__ == '__main__':
    Fire(filter_DROP)
