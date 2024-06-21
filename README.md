Welcome to `sci-BLOOM`, `sci-BLOOM` is an educational chatbot designed for EPFL STEM students. It is based on the BLOOM-1b7 model, fine-tuned on our custom datasets, improved towards answering multiple-choice questions, and further size-reduced via quantization.


## Usage
Our models can be imported with the standard HuggingFace methods. Visit our model repo on HF!
- [The fine-tuned model](https://huggingface.co/Veture/merged_dpo_model)
- [The quantized model](https://huggingface.co/Veture/merged_autoGPTQ_dpo/tree/main)



## Base Model
Our models are based on the [BLOOM-1b7 model](https://huggingface.co/bigscience/bloom-1b7), it is a publicly available LLM.


## Methods
Starting from the base model, we implemented **Supervised Fine-Tuning (SFT)**, **Direct Performance Optimization (DPO)**, accompanied with **Low Rank Adaption (LoRA)** and **quantization** to boost the training speed and reduce model size.

To optimize the model towards answering multiple-choice questions (MCQs), we add specific measures to parse the model outputs to single English capital letters representing the option of choice. See [here](https://github.com/Tachi-67/sci-BLOOM/blob/main/haolong/mcqa_parser.ipynb).

We trained our models on a VM with the NVIDIA-N1 GPU, and 16GB of CPU memory.


## Training Data
Our data for SFT and DPO are publicly avavilable on HuggingFace!
- [SFT Data](https://huggingface.co/datasets/Tachi67/sft_dataset)
- [DPO Data](https://huggingface.co/datasets/Tachi67/mnlp_dpo_data_7k)

## Results
We provide 2 versions of our fine-tuned models:
- [The fine-tuned model](https://huggingface.co/Veture/merged_dpo_model)
- [The quantized model](https://huggingface.co/Veture/merged_autoGPTQ_dpo/tree/main)


We used SFT and DPO (with LoRA) to fine-tune the model. After those, we applied quantization (GPTQ) on the fine-tuned model. For the base model, the SFT-ed model, the DPO-ed model, and the quantized model, we finally apply the MCQ parser to each of them and evaluate these models on labeled MCQs. Below are some facts from our evaluation:
- LoRA reduced **99.95%** of training parameters.
- SFT model's accuracy improved **20.6%** as compared to the base model.
- DPO model's accuracy improved **40.2%** as compared to the base model.
- Quantization reduced **67.1%** of the model size.
- Quantized model's accuracy only decreased **2%** as compared to the DPO model.

We conclude that our fine-tuning methods are significantly effective in terms of improving the model's ability to solve STEM questions. Meanwhile, the quantization we implemented largely reduces the model size with a promise of keeping the model's competence.

