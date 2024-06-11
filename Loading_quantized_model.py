from transformers import AutoModelForCausalLM
from peft import PeftModel
from quanto import freeze, qint8, quantize, safe_load
import torch

model_id = "bigscience/bloom-1b7"
dpo_path = "./m3_dpo"
out_path = './normal_quantized_m3'

# Load the base model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
print('Base model loaded.')

# Integrate LoRA Adapter
model = PeftModel.from_pretrained(model, dpo_path, is_trainable=True, dtype=torch.float16)
print('LoRA adapter integrated.')

# Ensure the model is quantized as expected
quantize(model)
print('Model quantized.')

# Load the quantized model state dictionary
quantized_state_dict = safe_load(out_path)
model.load_state_dict(quantized_state_dict)
print('Quantized model state dictionary loaded.')



# Verify the model is loaded correctly
print(model)
quantized_weights = {k: v for k, v in model.state_dict().items()}
print(quantized_weights)