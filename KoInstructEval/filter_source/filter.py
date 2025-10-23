"""
Script Name: filter.py
Author: Dongjun Kim
Affiliation: Korea University NLP & AI Lab
Email: junkim100@gmail.com
Description: This script is developed as part of the efforts described in the KoInstructEval paper,
             aimed at preparing English datasets for NLP tasks to be more accessible for Korean speakers.
             It filters out incongruent English data that lacks relevance or translatability to the Korean context
             and translates the remaining data into Korean, maintaining as much of the original context as possible.
             The script supports multiple datasets including MMLU, DROP, BBH, HumanEval, and IMPACT, providing
             functionalities for debugging and verbose output to facilitate detailed processing analysis.
License: Apache 2.0
Usage: 
    python filter.py --input_dir <input_directory> --output_dir <output_directory>
                     --dataset <dataset_names> [--debug] [--verbose]

Options:
    --input_dir: Specifies the directory where the input datasets are located.
    --output_dir: Specifies the directory where the filtered and translated datasets will be stored.
    --dataset: Identifies the dataset(s) to be processed, comma-separated (e.g., MMLU,DROP).
    --debug: Enables debugging mode for additional logging information.
    --verbose: Activates verbose output for more detailed operation logs.

Example:
    To filter and translate MMLU and DROP datasets with verbose output:
    python filter.py --input_dir ./data/source --output_dir ./data/filtered --dataset MMLU,DROP --verbose
"""
#%%
from filter_MMLU import filter_MMLU
from filter_DROP import filter_DROP
from filter_BBH import filter_BBH
from filter_HumanEval import filter_HumanEval
from filter_Impact import filter_Impact

import os
from fire import Fire

import numpy as np
np.float=float
from pororo import Pororo
#%%
SHOT_FILTER= {
    "MMLU": [
                {"role": "system", "content": "Assess whether the following questions and answers are relevant or irrelevant to a Korean audience, focusing on the translatability and comprehensibility of the content. Mark a prompt as relevant only if the concept can be fully understood by Korean speakers upon translation. If a concept relies on linguistic features not present in Korean or if the reference is culturally specific and wouldn't make sense to a Korean audience despite translation, mark it as irrelevant. The answer should only be 'relevant' or 'irrelevant', not the answer to the question itself"},
                {"role": "user", "content": "question: Calculate the volume of a sphere with a radius of 4 units.\nanswer: 268.08 cubic units"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: What holiday do Americans celebrate on the fourth Thursday of November?\nanswer: Thanksgiving"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: In Shakespeare's play, who says 'To be, or not to be, that is the question'?\nanswer: Hamlet"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Translate 'cat' into French.\nanswer: chat"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Who was the first president of the United States?\nanswer: George Washington"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Provide the past tense form of the irregular verb 'to fly'.\nanswer: flew"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Explain the significance of cherry blossoms in Japanese culture.\nanswer: Cherry blossoms, or 'Sakura,' symbolize the fleeting nature of life and are celebrated with viewing festivals called 'Hanami.'"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Name the chemical element with the atomic number 17.\nanswer: Chlorine"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Which planet is known as the Red Planet?\nanswer: Mars"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Who painted the Mona Lisa?\nanswer: Leonardo da Vinci"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Capitalize the first letter of 'seoul'.\nanswer: Seoul"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Identify the pronoun in the sentence: 'She is going to the market.'\nanswer: She"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Modify the sentence to passive voice: 'The committee approved the new policy.'\nanswer: The new policy was approved by the committee."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What's the plural form of 'cactus'?\nanswer: cacti"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Use semicolons to separate items in this complex list: 'We bought shiny, red apples; small, tart oranges; and large, green pears.'\nanswer: "},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Spell the word 'encyclopedia' backwards.\nanswer: aidepolcycne"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What is the rhyming scheme of the Shakespearean sonnet?\nanswer: ABABCDCDEFEFGG"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Correct the spelling: 'necesary'.\nanswer: necessary"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: In what tense is the sentence: 'I will have been running.'\nanswer: Future perfect continuous"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Provide an antonym for 'ancient'.\nanswer: Modern"},
                {"role": "assistant", "content": "irrelevant"}
            ],
    "DROP": [
                {"role": "system", "content": "Determine if the following question and answer pairs from the DROP dataset are relevant or irrelevant for Korean users. Mark a prompt as relevant only if the concept, when translated into Korean, can be understood by native Korean speakers, including but not limited to, mathematical reasoning, historical facts, and universal science concepts that are not culture-specific. Mark as irrelevant if the question or answer relies on conversational idioms, metaphors, or references not known or recognized in Korean culture, or if linguistic features do not have direct equivalents in Korean. The answer should only be 'relevant' or 'irrelevant', not the answer to the question itself"},
                {"role": "user", "content": "question: How many years between 1910 and 1950?\nanswer: 40 years"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: What year was 100 years before 2021?\nanswer: 1921"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Calculate the total score if a team scored 23, 34, and 19 points in three games.\nanswer: 76"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: If you have five apples and you give away two, how many do you have left?\nanswer: 3"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Who was the President of the United States in 1950?\nanswer: Harry S. Truman"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: On what date does the summer solstice occur?\nanswer: June 21"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: What is the freezing point of water?\nanswer: 0 degrees Celsius"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: If a train leaves Seoul at 3 PM and arrives in Busan at 7 PM, how long was the journey?\nanswer: 4 hours"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Translate this sentence into French: 'I love eating kimchi.'\nanswer: J'aime manger du kimchi."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Explain the significance of the bald eagle to American culture.\nanswer: It's the national bird and symbol of the USA."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What ingredients are needed to make a traditional American Thanksgiving dinner?\nanswer: Turkey, stuffing, mashed potatoes, cranberry sauce, and pumpkin pie."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Describe the taste of Marmite.\nanswer: "},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Using the metric system, convert 10 miles to kilometers.\nanswer: 16.09 kilometers"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Name the process by which plants make their food.\nanswer: Photosynthesis"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What are the colors of the rainbow?\nanswer: Red, orange, yellow, green, blue, indigo, violet."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Who wrote the national anthem of South Korea?\nanswer: Ahn Eak-tai"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: In what unit is electric current measured?\nanswer: Ampere"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What is the chemical symbol for water?\nanswer: H2O"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Describe a traditional game played during Korean New Year.\nanswer: Yut Nori"},
                {"role": "assistant", "content": "irrelevant"}
            ],
    "BBH": [
                {"role": "system", "content": "Determine if the following question and answer pairs from the BBH dataset are relevant or irrelevant for native Korean speakers. Mark a prompt as 'relevant' if, when translated into Korean, the concept remains understandable and would make sense to a Korean audience. Mark as 'irrelevant' only if the concept, upon translation, would become impossible to understand for a native Korean speaker due to linguistic or cultural incompatibilities. The answer should only be 'relevant' or 'irrelevant', not the answer to the question itself, not the answer to the question itself."},
                {"role": "user", "content": "question: If 'True AND False' is False, what is 'True AND True'?\nanswer: True"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: In English, 'to be or not to be' reflects a binary choice. Translate the expression into Binary.\nanswer: 01"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Does rain cause wet streets?\nanswer: Yes"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: If I become a poet, will the moon be bigger?\nanswer: No"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What month comes after June?\nanswer: July"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Translate 'July 4th Independence Day' to a Korean holiday equivalent.\nanswer: "},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: A bat can fly. What is it?\nanswer: An animal"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: 'Cricket' can refer to a game or an insect. Translate this ambiguity in Korean.\nanswer: "},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Add two to five, then multiply by three. What’s the result?\nanswer: 21"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Is 'multiplying before adding' the rule you should follow?\nanswer: No"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Is 'Happy Merry Christmas' a correct translation for '메리 크리스마스'?\nanswer: No"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: If object A is larger than B and B is larger than C, which is the smallest?\nanswer: C"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Considering seven objects, if G is heavier than F but lighter than E, and D is the heaviest, which one is in the middle?\nanswer: "},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Three cards are placed facedown: a King, a Queen, and a Jack. If the King is to the left of the Queen, and the Jack is to the right of the Queen, which card is in the middle?\nanswer: Queen"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Seven marbles are aligned, and the green marble is three places left to the red marble with a blue marble in between. If the yellow marble is to the far right, which marble is three places left to the yellow marble?\nanswer: "},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Is arguing that because many people believe in a rumor, it must be true, a logical fallacy?\nanswer: Yes"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: If an argument states 'All A are B, and some B are C, thus all A are C', is this a formal fallacy?\nanswer: Yes"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Rearrange the words for clarity: 'Gone to the market has she.'\nanswer: 'She has gone to the market.'"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Correct the order: 'Happy, I am today very.'\nanswer: 'I am very happy today.'"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: How many sides does a hexagon have?\nanswer: 6"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: If a triangle has two equal sides, what is it called?\nanswer: Isosceles triangle"},
                {"role": "assistant", "content": "relevant"}
            ],
    "IMPACT": [
                {"role": "system", "content": "For each provided question, determine if it's relevant or irrelevant for a Korean context. A question is considered relevant if its concept remains understandable to a native Korean speaker upon translation, irrespective of the original cultural context. Mark it irrelevant if the concept is impossible to understand by native speakers when translated, due to non-existence of the concept in Korean culture or language. The answer should only be 'relevant' or 'irrelevant', not the answer to the question itself"},
                {"role": "user", "content": "question: What is the process of electing a president in the United States?"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Discuss the cultural significance of wearing white at a wedding."},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Explain the term 'Fourth Estate' and its role in society."},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: What is the significance of the cherry blossom festival in Japan?"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Describe the historical impact of the Silk Road."},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Explain the rules of American football."},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Outline the principle behind separation of powers in government."},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Describe how Thanksgiving is celebrated."},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Compare and contrast the philosophical ideas of Confucius and Aristotle."},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: What is the significance of the Turing Test in artificial intelligence?"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Describe the impact of Brexit on the European Union."},
                {"role": "assistant", "content": "relevant"},    
                {"role": "user", "content": "question: What are the mechanics of a combustion engine?"},
                {"role": "assistant", "content": "relevant"},
                {"role": "user", "content": "question: Detail the importance of punting in American football."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Explain the tradition of Groundhog Day in America."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Describe the cultural implications of the right to bear arms in the United States."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Discuss the creation and meaning behind the Thanksgiving turkey pardon."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What is the origin of the term 'lunatic fringe'?"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Explain the tradition and rules of tossing a coin into the Trevi Fountain."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What are the narrative elements unique to Shakespearean comedies?"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Explain the significance of 'piñata' in Mexican celebrations."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Discuss the history and cultural significance of tarot cards."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: What is the concept of 'manifest destiny' in American history?"},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Explain the importance and symbolism of the Statue of Liberty."},
                {"role": "assistant", "content": "irrelevant"},
                {"role": "user", "content": "question: Describe the tradition and significance of the midnight sun in Norwegian culture."},
                {"role": "assistant", "content": "irrelevant"}
            ]
}

HELP_MESSAGE = """
    Usage: filter.py [OPTIONS]

    This script is designed to filter and process datasets for NLP tasks, allowing for the selection of specific datasets and the output of processed data. 

    Options:
    --input_dir <directory>     Specifies the directory where the input datasets are located.
                                Example: --input_dir ./source_dataset

    --output_dir <directory>    Specifies the directory where the filtered datasets should be saved.
                                Example: --output_dir ./output

    --dataset <name(s)>         Specifies which dataset(s) to process. Multiple datasets can be selected
                                and separated by a comma. Available datasets are: MMLU,DROP,BBH,HumanEval,IMPACT
                                Example: --dataset MMLU,DROP

    --debug                     Enables debug mode, providing more detailed error and processing messages
                                to assist with troubleshooting.

    --verbose                   Enables verbose mode, providing detailed messages about the script’s
                                operations to the user. If not set, the script will run in a quieter mode,
                                outputting minimal information.

    --help                      Shows this help message and exits.

    Examples:
    To filter the MMLU and DROP datasets with verbose output:
        python filter.py --input_dir ./source_dataset --output_dir ./output --dataset MMLU,DROP --verbose

    To run in debug mode for the HumanEval dataset:
        python filter.py --input_dir ./source_dataset --output_dir ./output --dataset HumanEval --debug

    Note: Ensure the paths provided for both the input and output directories exist and are accessible.
"""

def filter_dataset(input_dir='../source_dataset', output_dir='../output/', dataset=['MMLU', 'DROP', 'BBH', 'HumanEval', 'IMPACT'], debug=False, verbose=False, help=False):
    if help:
        print(HELP_MESSAGE)
        return
    if not os.path.exists(output_dir):
        print(f'Creating output directory: {output_dir}')
        os.makedirs(output_dir)
    translate_text = lambda x: Pororo(task="machine_translation", lang="multi")(x, src='en', tgt='ko')
    try:
        if 'MMLU' in dataset: filter_MMLU(os.path.join(output_dir,'MMLU'), os.path.join(input_dir,'MMLU'), SHOT_FILTER['MMLU'], translate_text, debug=debug, verbose=verbose)
        if 'DROP' in dataset: filter_DROP(os.path.join(output_dir,'DROP'), os.path.join(input_dir,'DROP'), SHOT_FILTER['DROP'], debug=debug, verbose=verbose)
        if 'BBH' in dataset: filter_BBH(os.path.join(output_dir,'BBH'), os.path.join(input_dir,'BBH'), SHOT_FILTER['BBH'], debug=debug, verbose=verbose)
        if 'HumanEval' in dataset: filter_HumanEval(os.path.join(output_dir,'HumanEval'), os.path.join(input_dir,'HumanEval'), debug=debug, verbose=verbose)
        if 'IMPACT' in dataset: filter_Impact(os.path.join(output_dir,'IMPACT'), os.path.join(input_dir,'IMPACT'), SHOT_FILTER['IMPACT'], debug=debug, verbose=verbose)
    except Exception as e:
        print(e)
        print(HELP_MESSAGE)

if __name__ == '__main__':
    Fire(filter_dataset)