from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
from typing import List
import re
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prep_input(input_sample:dict):
    # Extracting the question and options from the prompt
    # returns the instruction, question_body, and options
    prompt = input_sample['question'].replace("\nAnswer:", "")
    question_body = prompt.split('Options:')[0].strip()
    options = prompt.split('Options:')[1].strip()
    instruction = f"""Answer the following question in English:
{question_body}
Choose from the following options:
Options:\n{options}
    
Please provide only one answer with the letter of the option.
The chosen option is:"""
    return instruction, question_body, options

def generate_answer(instruction, model, tokenizer, max_new_tokens=42): # 42 because it's the mean of the lengths of the option texts (see below)
    # generate the answer to the question from the model
    input = tokenizer.encode(instruction, return_tensors="pt").to(device)
    output = model.generate(input, max_new_tokens=max_new_tokens)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_str

def extract_parts(text):
    # Extract the letter and the text of the option
    # Attempt to match the expected pattern "Letter. Text"
    match = re.match(r"([A-Z])\. (.*)", text)
    if match:
        return match.group(1), match.group(2).strip()
    else:
        # If no text follows the letter and period, handle gracefully
        if re.match(r"([A-Z])\.$", text.strip()):
            return text.strip()[0], ""
        else:
            print(f"Error while extracting part: {text}")
            raise ValueError("Error with extract parts")

def split_text_into_list(text):
    # Parse the options list from the options text
    items = re.split(r'(?m)^\s*([A-Z]\.)', text.strip())
    cleaned_items = []
    for i in range(1, len(items), 2):
        if i + 1 < len(items):
            cleaned_items.append(items[i] + items[i + 1])
    return cleaned_items


def parse_options(options:str):
    # Split the options into a list of tuples
    # Each tuple contains the letter and the text of the option
    options_list = split_text_into_list(options)
    res = []
    for option in options_list:
        if option:
            try:
                res.append(extract_parts(option.strip()))
            except:
                print("========")
                print("Error!!! Original text:\n",options)
                print("========")
    return res

def parse_mcqa_output(whole_text:str, instruction:str, options: str):
    """
    @param whole_text (str): the output string from the model, including the input string (instructions)
    @param instruction (str): the instruction string used to generate the output (input prompts)
    @param options (str): the options given in the question, also included in the instruction
    @return (str): a single capitalized letter (A-Z), the answer to the MCQ question, as extracted from the model output.
    
    Used to parse the plain text output from the model, and extract the MCQ answer from it.
    """    
    
    def find_high_priority_answer(model_generation, len_options):
        """If the model is outputing something like:
        D
        Explanation: .....
        or:
        D.
        Explanation: .....
        or:
        (d)
        Explanation: .....
        or:
        (D)
        Explanation: ......
        
        Then we can safely assume that the answer is D and can be searched with the alphabetical search function.
        """
        best_index = len(model_generation)
        best_char = None
        characters = [chr(ord('A') + i) for i in range(len_options)]
        for char in characters:
            index = model_generation.find(char)
            if index != -1 and index < best_index:
                best_index = index
                best_char = char
        
        # match (a) or (A) patterns
        if best_char is None:
            for char in characters:
                pattern = f"({char.lower()})"
                if pattern in model_generation:
                    best_char = char
                    best_index = model_generation.find(pattern)
                    break
                
        if model_generation[best_index+1:].startswith("\n") or model_generation[best_index+1:].startswith("."):
            return best_char
        return "GG"
        
    
    def find_alphabetical_answer(model_generation, len_options):
        """Find the first capital letter in the model output, and return it as the answer.
        It's not so reliable, but it's a good fallback if the high priority search fails.
        Failure cases (example):
        "The answer is D" -> will return T
        """
        
        best_index = len(model_generation)
        best_char = None
        characters = [chr(ord('A') + i) for i in range(len_options)]
        for char in characters:
            index = model_generation.find(char)
            if index != -1 and index < best_index:
                best_index = index
                best_char = char
        # match (a) patterns
        if best_char is None:
            for char in characters:
                pattern = f"({char.lower()})"
                if pattern in model_generation:
                    best_char = char
                    break
        if best_char is None:
            return "GG"
        return best_char
    
    # Below are content matching functions, to either match model geneation again the options, or the reverse.
    
    def clean_model_generation(model_generation):
        # cleaning the model output
        model_generation = model_generation.strip()
        # remove the first - if it exists
        if model_generation.startswith("-"):
            model_generation = model_generation[1:]
        # remove the starting 'Option x:' if it exists
        model_generation = re.sub(r"Option \d+: ", "", model_generation)
        return model_generation.strip()
    
    
    def match_option_content(model_generation, options_list):
        # match the options against the model generation
        # find the option that is closest to the start of the model generation, it's the most likely answer
        model_generation = clean_model_generation(model_generation)
        best_option = None
        best_index = len(model_generation)
        for option in options_list:
            match_pos = model_generation.find(option[1])
            if match_pos != -1 and best_index > match_pos:
                best_index = match_pos
                best_option = option[0]
        if best_option is None:
            return "GG"
        return best_option
    
    
    def match_option_content_reverse(model_generation, options_list):
        # match the model generation against the options
        model_generation = clean_model_generation(model_generation)
        for option in options_list:
            option_text = option[1]
            if model_generation in option_text:
                return option[0]
        return "GG"
    
    def dangerous_match(model_generation, options_list):
        # some model generations contain latex code that is not exactly what the options are
        # e.g. model generation = \emph{some text}, while the option = some text
        # it's a bit dangerous, because there are also math expressions in the context
        # so we use the match for the least priority
        def dangerous_clean(text):
            clean_text = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", text)
            return clean_text
        # clean the model generation
        model_generation = clean_model_generation(model_generation)
        model_generation = dangerous_clean(model_generation)
        # also clean the options
        new_options_list = options_list.copy()
        for idx, option in enumerate(new_options_list):
            # option[1] = dangerous_clean(option[1])
            new_options_list[idx] = (option[0], dangerous_clean(option[1]))
        # do the pos and reverse match for the dangerous cleaned text
        pos_match = match_option_content(model_generation, new_options_list)
        rev_match = match_option_content_reverse(model_generation, new_options_list)
        if pos_match != "GG":
            return pos_match
        elif rev_match != "GG":
            return rev_match
        else:
            return "GG"
    
    try:   
        model_generation = whole_text[len(instruction):] # remove the instruction part, only keep the answer
        options_list = parse_options(options)
        len_options = len(options_list)
        
        high_priority_search = find_high_priority_answer(model_generation, len_options)
        AB_search = find_alphabetical_answer(model_generation, len_options)
        content_search =  match_option_content(model_generation, options_list = options_list)
        reverse_content_serach =  match_option_content_reverse(model_generation, options_list = options_list)
        dangerous_search = dangerous_match(model_generation, options_list = options_list)
       
        # note the priority order
        if high_priority_search != "GG":
            return high_priority_search
        elif reverse_content_serach != "GG":
            return reverse_content_serach
        elif content_search != "GG":
            return content_search
        elif AB_search != "GG":
            return AB_search
        elif dangerous_search != "GG":
            return dangerous_search
        else:
            # all search failed, return a random guess
            return random.choice([chr(ord('A') + i) for i in range(len_options)])
    except:
        # something unexpected happened, return a random guess
        return random.choice([chr(ord('A') + i) for i in range(26)])

def get_mcqa_output(input_sample:dict, model, tokenizer):
    """Wrapper of all previous functions
    @param input_sample (dict): a dictionary containing the prompt of the question, the dict must have a key 'prompt', which contains the question and the 'Options' part.
    @param model (AutoModelForCausalLM): the model to use for the generation
    @param tokenizer (AutoTokenizer): the tokenizer to use for the generation
    @return (str): single capitalized letter (A-Z), the answer to the MCQ question.
    """
    instruction, _, options = prep_input(input_sample)
    output_str = generate_answer(instruction, model, tokenizer, max_new_tokens=42)
    return parse_mcqa_output(output_str, instruction, options)
        