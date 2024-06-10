from peft import AutoPeftModelForSequenceClassification
import torch
from transformers import AutoTokenizer

def merge_and_upload(adaptor_model_id, destination_repo_id, tokenizer = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model = AutoPeftModelForSequenceClassification.from_pretrained(adaptor_model_id)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)
    print("Model merged")
    merged_model.push_to_hub(destination_repo_id)
    print("Model uploaded")
    if tokenizer:
        tokenizer.push_to_hub(destination_repo_id)
        print("Tokenizer uploaded")
    

if __name__ == "__main__":
    adaptor_model_id = ...
    base_model_id = ... # the quantized base model
    destination_repo_id = "Tachi67/mnlp_dpo_model_bloom" # change it to your repo id
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    merge_and_upload(adaptor_model_id, destination_repo_id, tokenizer)
    print("All models uploaded!")