from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained('./lora_model')
tokenizer = AutoTokenizer.from_pretrained('./lora_model')

# Push to the hub
model.push_to_hub('emirkocer/mistral-7b-v03-finetuned-math-v2')
tokenizer.push_to_hub('emirkocer/mistral-7b-v03-finetuned-math-v2')
