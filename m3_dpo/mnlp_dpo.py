#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
#from safetensors.torch import save_file, load_file
import torch.nn.utils.prune as prune
import smoothquant



indexes_to_remove = [1523]


def filter_by_index(example, idx):
    return idx not in indexes_to_remove


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print device info
print("Using device:", device)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# load dataset
dataset = load_dataset("Tachi67/mnlp_dpo_data_7k")
dataset = dataset.filter(filter_by_index, with_indices=True)
train_dataset = dataset['train'].filter(filter_by_index, with_indices=True)
eval_dataset = dataset['test'].filter(filter_by_index, with_indices=True)
#train_dataset = dataset['train']
#eval_dataset = dataset['test']

#################
# Experiment    #
#################
sample_size = 100
small_train_dataset = train_dataset.select(range(sample_size))
small_eval_dataset = dataset['test'].select(range(sample_size))


model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# add padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter
dpo_path = "./sft_test"
model = PeftModel.from_pretrained(model, dpo_path, is_trainable=True, dtype=torch.float16)


model.enable_input_require_grads()
#model.print_trainable_parameters()
model.to(device)

# set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="no",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=100,
    learning_rate=3e-4,
    weight_decay=0.05,
    logging_dir='./logs',
    logging_steps=50,
    fp16=True,
    fp16_opt_level='O1',
    remove_unused_columns=False,
    gradient_accumulation_steps=16,
)

# initialize Trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    #train_dataset=small_train_dataset,
    eval_dataset=eval_dataset,
    #eval_dataset=small_eval_dataset,
    max_length=512,
    max_prompt_length=512,
    #peft_config=adapter_config,
    #peft_config=new_lora_config
)

print(torch.cuda.memory_summary())

# check trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

print("Part of the model weights：")
for name, param in model.named_parameters():
    if 'lora' in name:
        print(name, param.data)

torch.cuda.empty_cache()

# training
trainer.train()

# save model
model_output_dir = './saved_model'
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
model.save_pretrained("./best_model_dpo")



