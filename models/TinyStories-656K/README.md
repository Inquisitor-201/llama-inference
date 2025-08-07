---
license: apache-2.0
widget:
- text: '<|start_story|>Once upon a time, there was a little boy named Tim. Tim '
  example_title: Sample 1
datasets:
- raincandy-u/TinyStoriesV2_SpecialTokens
language:
- en
library_name: transformers
---

# TinyStories-656K

This is a LM trained from scratch on TinyStoriesV2 dataset.

Aims to be a transformer language model capable of generating story with only 600k~ of parameters.

- Llama Architecture
- GQA
- hidden_size = 128
- Use tie_word_embeddings
- vocab_size=2048 (Trained on TinystoriesV2 from scratch, using BPE)
- 2 Transformers Layers

Code: [Here](https://github.com/Ce-daros/Tinystory-LM)

## Full Training Arguments

```
training_args = TrainingArguments(
  do_train=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=1,
  learning_rate=0.004629403549377777,
  lr_scheduler_type="constant",
  bf16=True,
  logging_steps=5,
  num_train_epochs=2,
  save_steps=10000000,
  seed=3407,report_to=None
)
```

# Generation

Template:

```
<|start_story|>Once upon a time, 
```

Generation example:

```
Once upon a time, there was a little boy named Tim. Tim had a toy car that he loved to play with. One day, he went to the park with his mom. Tim saw a toy car on the ground. Tim wanted to play with the car to his mom and said, "Mom, can I play with your car with my car too?"
His mom said, "Yes, but we must not take turns." Tim felt sad, but he knew he had to go. He asked his mom for help. His mom said, "Okay, let's clean it together." They went to play together and played the toy car. They had a lot of fun.
After they finished the car together, Tim and his mom were surprised. They did not know that the car was not a toy car like it was a magic car. Tim had an idea. He put the car in the car and put the car on it. He pushed the car on the car on the car car and pulled it down. Tim was so happy. He played with the car with his car all day long, and Tim was very happy.<|end_story|>
```

Recommended generation config:

```
do_sample=True,
top_k=40,
top_p=0.9,
temperature=0.6
```