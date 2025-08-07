from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

tokenizer: transformers.PreTrainedTokenizer = AutoTokenizer.from_pretrained("models/TinyStories-656K")

prompt = "one "
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(input_ids)
