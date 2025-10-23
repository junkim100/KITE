import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatasetProcessor:
    def __init__(self, dataset_name, dataset_subset, dataset_split, output_file, question_key, context_key, additional_keys=None, stop=100):
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split
        self.output_file = output_file
        self.question_key = question_key
        self.context_key = context_key
        self.additional_keys = additional_keys if additional_keys else []
        self.stop = stop
        self.dataset = load_dataset(self.dataset_name, self.dataset_subset, split=self.dataset_split)

    def normalize_text(self, text):
        return ' '.join(text.lower().strip().split())

    def process(self):
        data_set = set()

        # Iterate through the dataset
        if self.context_key == None:
            for data in self.dataset:
                question = self.normalize_text(data[self.question_key])
                data_set.add(question)
        else:
            for data in self.dataset:
                context = self.normalize_text(data[self.context_key])
                question = self.normalize_text(data[self.question_key])
                concatenated_string = context + " " + question
                data_set.add(concatenated_string)

        print(f"Number of unique {self.dataset_name} prompts: {len(data_set)}")

        cur_idx = 0

        # Open the file in write mode initially to clear it or create it if it doesn't exist
        with open(self.output_file, "w", encoding="utf-8") as file:
            file.truncate(0)

        # Open the file in append mode for subsequent writing
        with open(self.output_file, "a", encoding="utf-8") as file:
            for input_data in data_set:
                if cur_idx == self.stop:
                    break
                if cur_idx % 10 == 0:
                    print(f"Processed {cur_idx}th prompt")

                response = (
                    client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "text": "당신은 한국어만 구사하고 이해할 수 있습니다. 당신의 임무는 Q&A 데이터를 지침 따르기 데이터로 변환하는 것입니다. 모든 입력은 Q&A 데이터이며, Q&A 데이터 자체에 의존하지 않고 이를 기반으로 자세하고 창의적인 지시문 프롬프트를 작성해야 합니다. 작성된 지시문은 원래 Q&A 데이터를 보지 않고도 따를 수 있어야 합니다.",
                                        "type": "text"
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "한국의 수도는 어디인가요?",
                                        "type": "text"
                                    }
                                ]
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "text": "한국의 수도로서 서울의 중요성을 설명하세요.",
                                        "type": "text"
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "현재 한국의 대통령은 누구입니까?",
                                        "type": "text"
                                    }
                                ]
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "text": "현재 한국 대통령의 주요 정책에 대한 간략한 개요를 말해주세요.",
                                        "type": "text"
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": input_data,
                                        "type": "text"
                                    }
                                ]
                            }
                        ],
                        temperature=1,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    ).choices[0].message.content
                )

                # Create a dictionary combining the original data and the response
                response_data = {
                    "instruction": response,
                    "original_data": input_data
                }

                # Write the dictionary to the file in JSONL format
                file.write(json.dumps(response_data, ensure_ascii=False) + "\n")
                file.flush()  # Ensure data is written to the file
                cur_idx += 1

        print(f"Finished processing {self.dataset_name} dataset")

if __name__ == "__main__":
    datasets = [
        # {
        #     "name": "naver-ai/kobbq",
        #     "subset":  None,
        #     "split": "test",
        #     "output_file": "../data/qa_to_if/kobbq.jsonl",
        #     "question_key": "question",
        #     "context_key": "context"
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Accounting",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Accounting.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Agricultural-Sciences",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Agricultural-Sciences.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Aviation-Engineering-and-Maintenance",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Aviation-Engineering-and-Maintenance.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Biology",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Biology.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Chemical-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Chemical-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Chemistry",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Chemistry.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Civil-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Civil-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Computer-Science",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Computer-Science.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Construction",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Construction.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Criminal-Law",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Criminal-Law.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Ecology",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Ecology.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Economics",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Economics.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Education",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Education.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Electrical-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Electrical-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Electronics-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Electronics-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Energy-Management",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Energy-Management.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Environmental-Science",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Environmental-Science.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Fashion",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Fashion.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Food-Processing",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Food-Processing.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Gas-Technology-and-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Gas-Technology-and-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Geomatics",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Geomatics.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Health",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Health.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Industrial-Engineer",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-.Industrial-Engineerjsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Information-Technology",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Information-Technology.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Interior-Architecture-and-Design",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Interior-Architecture-and-Design.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Korean-History",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Korean-History.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Law",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Law.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Machine-Design-and-Manufacturing",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Machine-Design-and-Manufacturing.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Management",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Management.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Maritime-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Maritime-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Marketing",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Marketing.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Materials-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Materials-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Math",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Math.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Mechanical-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Mechanical-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Nondestructive-Testing",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Nondestructive-Testing.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Patent",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Patent.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        {
            "name": "HAERAE-HUB/KMMLU",
            "subset":  "Political-Science-and-Sociology",
            "split": "train",
            "output_file": "../data/qa_to_if/kmmlu-Political-Science-and-Sociology.jsonl",
            "question_key": "question",
            "context_key": None
        }
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Psychology",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Psychology.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Public-Safety",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Public-Safety.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Railway-and-Automotive-Engineering",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Railway-and-Automotive-Engineering.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Real-Estate",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Real-Estate.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Refrigerating-Machinery",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Refrigerating-Machinery.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Social-Welfare",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Social-Welfare.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Taxation",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Taxation.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # },
        # {
        #     "name": "HAERAE-HUB/KMMLU",
        #     "subset":  "Telecommunications-and-Wireless-Technology",
        #     "split": "train",
        #     "output_file": "../data/qa_to_if/kmmlu-Telecommunications-and-Wireless-Technology.jsonl",
        #     "question_key": "question",
        #     "context_key": None
        # }
    ]

    for dataset_info in datasets:
        processor = DatasetProcessor(
            dataset_name=dataset_info["name"],
            dataset_subset=dataset_info["subset"],
            dataset_split=dataset_info["split"],
            output_file=dataset_info["output_file"],
            context_key=dataset_info["context_key"],
            question_key=dataset_info["question_key"],
            # additional_keys=dataset_info["additional_keys"]
        )
        processor.process()