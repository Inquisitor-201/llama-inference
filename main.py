# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("raincandy-u/TinyStories-656K")
# model = AutoModelForCausalLM.from_pretrained("raincandy-u/TinyStories-656K")

tokenizer = AutoTokenizer.from_pretrained("./models/TinyStories-656K")
model = AutoModelForCausalLM.from_pretrained("./models/TinyStories-656K")

# Generate text
# prompt = "One day, a girl named "
prompt = "One day, a girl named Lily went for a "
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# generated_ids = model.generate(input_ids=input_ids, max_length=100, do_sample=True, top_p=0.95, top_k=60)
# generated_ids = model.generate(input_ids=input_ids, max_length=100, do_sample=False, top_p=0.95, top_k=60)
generated_ids = model.generate(input_ids=input_ids, max_length=200, do_sample=False)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(input_ids)
print(generated_ids)
print(generated_text)