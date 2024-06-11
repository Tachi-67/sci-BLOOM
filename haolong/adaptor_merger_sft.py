from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer

def merge_and_upload(adaptor_model_id, destination_repo_id, tokenizer = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model = AutoPeftModelForCausalLM.from_pretrained(adaptor_model_id)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)
    print("Model merged")
    merged_model.push_to_hub(destination_repo_id)
    print("Model uploaded")
    if tokenizer:
        tokenizer.push_to_hub(destination_repo_id)
        print("Tokenizer uploaded")
    

if __name__ == "__main__":
    adaptor_model_id = "Tachi67/mnlp_sft_model_bloom"
    
    # TODO: change it to your repo id
    destination_repo_id = "" # change it to your repo id
    # tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    merge_and_upload(adaptor_model_id, destination_repo_id, tokenizer = None)
    print("All models uploaded!")