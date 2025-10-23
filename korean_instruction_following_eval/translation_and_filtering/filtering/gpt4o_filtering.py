import argparse
import requests
import json
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if __name__ == "__main__":
    with open("../data/translated/translated.jsonl", "r") as file, open(
        "../data/filter/relevant.jsonl", "w"
    ) as relevant_file, open("../data/filter/irrelevant.jsonl", "w") as irrelevant_file:
        # Clear the contents of the output files
        relevant_file.truncate(0)
        irrelevant_file.truncate(0)

        line_number = 0
        responses = []
        retry_count = 0
        max_retries = 3
        for line in file:
            while True:
                data = json.loads(line)
                input_data = data["prompt"]

                response = (
                    client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "text": "당신은 한국어만 구사하고 이해 할 수 있습니다. 당신은 영어에서 한국어로 번역된 프롬프트를 분류하는 임무를 맡고 있습니다. 번역으로 인한 의미의 완전한 상실로 전혀 이해할 수 없게 되는 경우, 해당 프롬프트는 '관련성 없음(0)'으로 분류되어야 합니다. 예를 들어, 번역 과정에서 원본의 의미가 소실되거나, 번역에 오류가 명백하거나, 한국어에 존재하지 않는 당신이 이해 할 수 없는 언어적 특성, 문법 구조, 또는 관용구 등으로 인해 프롬프트의 의도가 왜곡되는 경우가 0에 해당합니다. 영어 문구가 포함된다고 무조건 관련성이 없는것은 아닙니다.\n나머지 이해 가능한 프롬프트는 '관련성 있음(1)'으로 분류되어야 합니다. 대부분의 프롬프트는 한국 청중에게 의미가 있고, 이해할 수 있을 것입니다. 여기에는 일반적인 지식, 논리적 추론, 과학적 사실, 그리고 한국어어로 설명 가능한 타문화 개념들 또한 포함됩니다. 이와 같은 프롬프트는 한국어로 번역되어도 의미와 가치를 유지하기 때문에 관련성이 있습니다.\n목표는 번역된 프롬프트가 한국 청중에게 실제로 의미 있게 전달될 수 있는지를 판단하는 것입니다. 대부분의 프롬프트가 이해 가능하고 문화적으로 관련이 있는 내용(1)이라는 점을 기억하세요.\n 오직 0이나 1로만 답변하세요.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "피타고라스 정리는 무엇에 관한 것인가요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "물의 화학식은 무엇인가요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "태양계에서 몇 번째 행성이 지구인가요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "비타민 C는 어떤 질병을 예방하는 데 도움이 되나요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "유네스코 세계 문화 유산 중 하나인 경복궁은 어느 도시에 위치해 있나요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "세계에서 가장 긴 강은 무엇인가요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "'안녕하세요'를 전부 대문자로 바꿔줘요.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "0", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "'피자'와 '파스타'는 어느 나라 음식인가요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "'Articles' (a, an, the)를 문장에서 지워주세요.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "0", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "샴페인이 원산지로 알려진 프랑스의 지역은 어디인가요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "'사파이어'는 어떤 종류의 광물인가요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "모든 A가 B이고 일부 B가 C이므로 모든 A가 C라는 논리가 제시되었다면, 이것은 형식적 오류입니까?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "옥스포드 대학은 어느 나라에 위치해 있나요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "이 문장에서 악센트 기호를 전부 없애주세요.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "0", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "당신이 사과 다섯 개를 가지고 있고 그 중 두 개를 주면, 몇 개가 남습니까?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "세계에서 가장 높은 빌딩 '부르즈 할리파'는 어느 도시에 위치하고 있나요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "일본 전통 음료인 '사케'는 어떤 재료로 만들어지나요?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "올림피아의 신전이 위치한 그리스의 도시는 어디인지 재미있게 말해줘요.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "미국의 첫 번째 대통령은 누구였습니까?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "개발 도상국에서 교육의 질을 향상시키기 위해 기계 학습과 AI를 어떻게 사용할지에 대한 프로젝트 제안서를 작성하십시오. 답변에서 쉼표를 사용하지 마십시오.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "2021년보다 100년 전은 어느 해인가?",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "'ancient'의 반의어를 제공하시오.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "0", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "모나리자를 그린 화가는 누구인가요? 전체 응답을 큰 따옴표로 감싸세요.",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"text": "1", "type": "text"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "윌리엄 셰익스피어의 '햄릿'에서 '살아야 할까, 죽어야 할까'라는 유명한 대사를 한 인물은 누구인지 마크다운으로 대답하세요. 예시 ```",
                                        "type": "text",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "1"}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "광합성 과정에서 식물은 어떤 가스를 흡수하고 어떤 가스를 방출하나요?",
                                    }
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "1"}],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": input_data}],
                            },
                        ],
                        temperature=1,
                        max_tokens=1,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    .choices[0]
                    .message.content
                )

                try:
                    response = int(response)
                    # print(f"Response {line_number}: {response}")
                except Exception as e:
                    if retry_count > max_retries:
                        print(
                            f"Max retries reached for line {line_number}. Skipping..."
                        )
                        retry_count = 0
                        break
                    print(f"Exception on line {line_number}: {str(e)}")
                    retry_count += 1
                    continue

                if response:
                    data["relevancy"] = response
                    relevant_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    data["relevancy"] = response
                    irrelevant_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                line_number += 1
                break

        # close the file
        file.close()
        relevant_file.close()
        irrelevant_file.close()
