import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
import os
import torch
import requests

load_dotenv()

SHOTS1 = [
    {
        "input": "두 가지 다른 방법으로 '나는 빨간색을 좋아합니다'를 표현해보세요. 두 가지 답변을 제공하고, 답변 사이에 별표 여섯 개(******)를 넣어 구분하세요.",
        "output": "나는 빨간색을 좋아해요.\n******\n내가 좋아하는 색은 빨간색이에요.",
    }
]

# SHOTS1 = [
#     {
#         "input": "조사는 '가', '이', '은', '는', '을', '를', '에', '에서', '와', '과', '로', '으로' 와 같이 체언이나 부사, 어미 따위에 붙어 그 말과 다른 말과의 문법적 관계를 표시하거나 그 말의 뜻을 도와주는 단어입니다. 조사를 사용하지 않고 지하철을 타는 법을 설명하세요.",
#         "output": "지하철역 도착, 지하철역 입구 찾기, 티켓 구매 또는 교통카드 준비, 개찰구 통과, 승강장 도착, 전광판 확인하여 열차 도착 시간 파악, 열차 도착 후 문 열리면 승차, 목적지 도착 시 하차, 개찰구 통과하여 역 밖으로 나오기",
#     }
# ]

SHOTS3 = [
    {
        "input": "두 가지 다른 방법으로 '나는 빨간색을 좋아합니다'를 표현해보세요. 두 가지 답변을 제공하고, 답변 사이에 별표 여섯 개(******)를 넣어 구분하세요.",
        "output": "나는 빨간색을 좋아해요.\n******\n내가 좋아하는 색은 빨간색이에요.",
    },
    {
        "input": "이번 주에 발표할 새로운 제품에 대한 광고를 작성하세요. 제목은 이중 꺾쇠 괄호로 감싸야 하며, 예를 들어, <<제목>>입니다. 답변에는 쉼표를 사용하지 마세요.",
        "output": "<<새로운 제품 출시>>\n이번 주에는 새로운 제품을 출시합니다. 이 제품은 많은 사람들이 기다리고 있는 제품이며, 많은 사람들이 기대하고 있습니다. ",
    },
    {
        "input": "영화 '엘카미노'에 대한 감상평을 작성하세요.\n먼저 요청한 내용을 단어 그대로 반복한 후에 답변하세요 (1. 요청을 반복하기 전에 어떤 단어나 문자도 말하지 마세요; 2. 반복해야 할 요청에는 이 문장이 포함되지 않습니다)",
        "output": "영화 '엘카미노'에 대한 감상평을 작성하세요.\n엘카미노는 브레이킹 배드의 후속작으로, 주인공 제시 플레먼이 월터 화이트의 사건 이후의 이야기를 다룹니다. 이 영화는 브레이킹 배드 팬들에게는 반가운 소식이지만, 그렇지 않은 사람들에게는 그다지 흥미로운 영화가 아닐 수 있습니다.",
    },
]

# SHOTS3 = [
#     {
#         "input": "조사는 '가', '이', '은', '는', '을', '를', '에', '에서', '와', '과', '로', '으로' 와 같이 체언이나 부사, 어미 따위에 붙어 그 말과 다른 말과의 문법적 관계를 표시하거나 그 말의 뜻을 도와주는 단어입니다. 조사를 사용하지 않고 지하철을 타는 법을 설명하세요.",
#         "output": "지하철역 도착, 지하철역 입구 찾기, 티켓 구매 또는 교통카드 준비, 개찰구 통과, 승강장 도착, 전광판 확인하여 열차 도착 시간 파악, 열차 도착 후 문 열리면 승차, 목적지 도착 시 하차, 개찰구 통과하여 역 밖으로 나오기",
#     },
#     {
#         "input": "다음 문장의 숫자를 한자어로 바꾸세요. '우리는 내일 아침 일곱 시에 출발할 거야.'",
#         "output": "우리는 내일 아침 칠 시에 출발할 거야.",
#     },
#     {
#         "input": "이행시는 주어진 두 글자 단어의 각 글자로 시작하는 일관성 있는 짧막한 이야기를 의미합니다. '로션'로 이행시를 지으세요. 답변은 반드시 단어에 포함되어 있는 각 글자로 시작해야 합니다. 답변에 '뜨거운'라는 단어를 최소 한 번 포함시키세요.",
#         "output": "로: 로맨틱한 저녁, 바닷가에서 션: 선물 받은 뜨거운 마음이 전해진다.",
#     },
# ]

SHOTS5 = [
    {
        "input": "두 가지 다른 방법으로 '나는 빨간색을 좋아합니다'를 표현해보세요. 두 가지 답변을 제공하고, 답변 사이에 별표 여섯 개(******)를 넣어 구분하세요.",
        "output": "나는 빨간색을 좋아해요.\n******\n내가 좋아하는 색은 빨간색이에요.",
    },
    {
        "input": "이번 주에 발표할 새로운 제품에 대한 광고를 작성하세요. 제목은 이중 꺾쇠 괄호로 감싸야 하며, 예를 들어, <<제목>>입니다. 답변에는 쉼표를 사용하지 마세요.",
        "output": "<<새로운 제품 출시>>\n이번 주에는 새로운 제품을 출시합니다. 이 제품은 많은 사람들이 기다리고 있는 제품이며, 많은 사람들이 기대하고 있습니다. ",
    },
    {
        "input": "영화 '엘카미노'에 대한 감상평을 작성하세요.\n먼저 요청한 내용을 단어 그대로 반복한 후에 답변하세요 (1. 요청을 반복하기 전에 어떤 단어나 문자도 말하지 마세요; 2. 반복해야 할 요청에는 이 문장이 포함되지 않습니다)",
        "output": "영화 '엘카미노'에 대한 감상평을 작성하세요.\n엘카미노는 브레이킹 배드의 후속작으로, 주인공 제시 플레먼이 월터 화이트의 사건 이후의 이야기를 다룹니다. 이 영화는 브레이킹 배드 팬들에게는 반가운 소식이지만, 그렇지 않은 사람들에게는 그다지 흥미로운 영화가 아닐 수 있습니다.",
    },
    {
        "input": '만약 x + y + z = 50이고 y = 15이며 z = 20라면, x = 15인가요? 다음 옵션 중 하나로 답하세요: "제 대답은 예입니다.", "제 대답은 아니오입니다.", "제 대답은 아마도입니다."',
        "output": '"제 대답은 예입니다."',
    },
    {
        "input": "친구에게 미안하다는 내용의 문자를 써주세요. 끝에 'P.S.'로 시작하는 추가사항을 넣어주세요.",
        "output": "안녕 [친구 이름],\n\n최근에 내가 했던 행동에 대해 정말 미안해. 너에게 상처를 줘서 정말로 죄송하게 생각해. 나도 그때 너무 흥분해서 생각 없이 말한 것 같아. 네 입장을 충분히 이해하지 못한 것 같아서 더욱 마음이 아파. 정말로 사과하고 싶어.\n\n얼굴 보고 사과하고 싶었지만, 차마 용기가 안 나서 이렇게 문자로 대신해. 너와의 소중한 우정을 다시 회복하고 싶어. 시간 내서 나와 이야기할 수 있다면 정말 고맙겠어.\n\nP.S. 네가 부탁한 책, 이번 주 내로 꼭 돌려줄게. 미안해서 계속 미루고 있었어. 다시 한 번 정말 미안해.",
    },
]

# SHOTS5 = [
#     {
#         "input": "조사는 '가', '이', '은', '는', '을', '를', '에', '에서', '와', '과', '로', '으로' 와 같이 체언이나 부사, 어미 따위에 붙어 그 말과 다른 말과의 문법적 관계를 표시하거나 그 말의 뜻을 도와주는 단어입니다. 조사를 사용하지 않고 지하철을 타는 법을 설명하세요.",
#         "output": "지하철역 도착, 지하철역 입구 찾기, 티켓 구매 또는 교통카드 준비, 개찰구 통과, 승강장 도착, 전광판 확인하여 열차 도착 시간 파악, 열차 도착 후 문 열리면 승차, 목적지 도착 시 하차, 개찰구 통과하여 역 밖으로 나오기",
#     },
#     {
#         "input": "다음 문장의 숫자를 한자어로 바꾸세요. '우리는 내일 아침 일곱 시에 출발할 거야.'",
#         "output": "우리는 내일 아침 칠 시에 출발할 거야.",
#     },
#     {
#         "input": "이행시는 주어진 두 글자 단어의 각 글자로 시작하는 일관성 있는 짧막한 이야기를 의미합니다. '로션'로 이행시를 지으세요. 답변은 반드시 단어에 포함되어 있는 각 글자로 시작해야 합니다. 답변에 '뜨거운'라는 단어를 최소 한 번 포함시키세요.",
#         "output": "로: 로맨틱한 저녁, 바닷가에서 션: 선물 받은 뜨거운 마음이 전해진다.",
#     },
#     {
#         "input": "다음 문장을 존댓말로 바꾸세요. '내가 준비한 서프라이즈가 있어. 너를 위한 선물이야.'",
#         "output": "제가 준비한 서프라이즈가 있습니다. 당신을 위한 선물입니다.",
#     },
#     {
#         "input": "조사는 '가', '이', '은', '는', '을', '를', '에', '에서', '와', '과', '로', '으로' 와 같이 체언이나 부사, 어미 따위에 붙어 그 말과 다른 말과의 문법적 관계를 표시하거나 그 말의 뜻을 도와주는 단어입니다. 조사를 사용하지 않고 현충일에 대해 설명하세요.",
#         "output": "매년 6월 6일에 거행, 나라 위해 목숨 바친 이들 추모, 애국심 고취 위한 날, 전국 곳곳서 기념식 열림, 태극기 반기로 게양, 묵념 통해 감사와 존경 표현, 전쟁과 관련된 장소 방문 많음, 호국영령 기억, 나라 사랑 마음 다짐의 날",
#     },
# ]


class Model:
    def __init__(self, model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.load_model_from_api()

    def process_prompts(self, input_file, output_file, shot_num):
        if shot_num == 0:
            shots = None
        elif shot_num == 1:
            shots = SHOTS1
        elif shot_num == 3:
            shots = SHOTS3
        elif shot_num == 5:
            shots = SHOTS5
        else:
            raise ValueError("Invalid shot_num. Choose 1, 3, or 5.")
        data = self.read_input_file(input_file)
        responses = []
        for entry in data:
            print(f"{self.model_name}-{shot_num}shot Entry {data.index(entry) + 1}...")
            instruction = entry["instruction"]
            response = self.generate(instruction, shots)
            out_data = {"instruction": instruction, "response": response}
            responses.append(out_data)
        self.save_response(responses, output_file)

    def save_response(self, responses, output_file):
        with open(output_file, "w", encoding="utf-8") as file:
            file.truncate(0)
        with open(output_file, "a", encoding="utf-8") as file:
            for response in responses:
                file.write(json.dumps(response, ensure_ascii=False) + "\n")

    def read_input_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        return data

    def load_model_from_api(self):
        raise NotImplementedError("This method should be overridden in the subclass")

    def generate(self, instruction, shots):
        raise NotImplementedError("This method should be overridden in the subclass")


class HuggingFaceModel(Model):
    def __init__(self, model_name):
        super().__init__(model_name)

    def load_model_from_api(self):
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_api_key:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=hf_api_key
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, token=hf_api_key
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def generate(self, instruction, shots):
        if shots:
            # Combine the shots with the instruction
            shots_text = "\n".join(
                [f"Input: {shot['input']}\nOutput: {shot['output']}" for shot in shots]
            )
            instruction = f"{shots_text}\nInput: {instruction}\nOutput:"

        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,  # Specify the maximum number of new tokens to generate
            pad_token_id=self.tokenizer.eos_token_id,  # Ensure proper padding
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class OpenAIModel(Model):
    def __init__(self, model_name):
        super().__init__(model_name)

    def load_model_from_api(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, instruction, shots):
        messages = []
        if shots:
            # Combine the shots with the instruction
            for shot in shots:
                messages.append({"role": "user", "content": shot["input"]})
                messages.append({"role": "assistant", "content": shot["output"]})

        # Add the actual instruction
        messages.append({"role": "user", "content": instruction})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=1,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content


class SolarAPI(OpenAIModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def load_model_from_api(self):
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar",
        )

    def generate(self, instruction, shots):
        return super().generate(instruction, shots)


class HyperCLOVA_X(Model):
    class CompletionExecutor:
        def __init__(self, host, api_key, api_key_primary_val, request_id):
            self._host = host
            self._api_key = api_key
            self._api_key_primary_val = api_key_primary_val
            self._request_id = request_id

        def execute(self, completion_request):
            headers = {
                "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
                "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
                "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "text/event-stream",
            }

            with requests.post(
                self._host + "/testapp/v1/chat-completions/HCX-003",
                headers=headers,
                json=completion_request,
                stream=True,
            ) as r:
                for line in r.iter_lines():
                    last_message = ""
                    for line in r.iter_lines():
                        if line:
                            # Decode the line and strip any leading/trailing spaces
                            decoded_line = line.decode("utf-8").strip()

                            # Check if the line starts with 'data:' and extract the JSON part
                            if decoded_line.startswith("data:"):
                                json_data = decoded_line[len("data:") :].strip()

                                # Parse the JSON data
                                try:
                                    data = json.loads(json_data)
                                    if (
                                        "message" in data
                                        and "content" in data["message"]
                                    ):
                                        last_message = data["message"]["content"]
                                except json.JSONDecodeError:
                                    # Handle possible JSON decode errors
                                    continue

                    # Return the final output messagee)
                    return last_message

    def __init__(self, model_name):
        super().__init__(model_name)

    def load_model_from_api(self):
        self.completion_executor = self.CompletionExecutor(
            host="https://clovastudio.stream.ntruss.com",
            api_key=os.getenv("HYPERCLOVAX_API_KEY"),
            api_key_primary_val="DQVncHs0pafi5LNkdqsLMvTDwpjdcgndq3R45rP6",
            request_id="03540065-357b-4e0f-86ab-4a34110e92a5",
        )

    def generate(self, instruction, shots):
        messages = []
        if shots:
            # Combine the shots with the instruction
            for shot in shots:
                messages.append({"role": "user", "content": shot["input"]})
                messages.append({"role": "assistant", "content": shot["output"]})

        # Add the actual instruction
        messages.append({"role": "user", "content": instruction})

        request_data = {
            "messages": messages,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 500,
            "temperature": 0.5,
            "repeatPenalty": 5.0,
            "stopBefore": [],
            "includeAiFilters": False,
            "seed": 0,
        }

        return self.completion_executor.execute(request_data)
