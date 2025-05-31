torch 2.5.1 cuda 12.1 is needed

**It seems we are not allowed to use chat templates so it is not possible to do instruction finetuning**

## Ideas
- Can we still train with the instruction datasets? or does it make sense to do it?# mnlp_mcqa_model
  - The answer is yes
- How much do we clip the grad norms, how much do we clip the grad norm of the linear layers
- how much gradient batch accumulation do we use
- should we use weight decay?
  - I think no
- What dataset to use for evaluation during training
- Try and use accelerate


## Notes
- When using a dataset with a "messages" field (like the example above), the SFTTrainer automatically applies the model's chat template, which it retrieves from the hub. This means you don't need any additional configuration to handle chat-style conversations - the trainer will format the messages according to the model's expected template format.
  - **So what we have to do for formatting is do {Question} \n {Answer} (the answer is yes it seems), and we should use packing when finetuning**
  - We have to shuffle the training set before each epoch
- Use Flash attention and accelerate
- Maybe use QLora
- Does SFTT and in general when doing finetuning, is the loss done on Question and Answer? or just the answer?
  - For the training also doing the masking of the question seems to depend on the finetuning task, for the instruction finetuning it seems we don't have to mask the question
    - Still not sure if we have to mask the question (or instruction) or not during training?
- Use packing maybe, because it helps putting multiple examples in a single prompt up to max seq length (packing adds eos sentence tokens between examples)
- We should use an effective batch size of 128
  - Doing small batch sizes makes the gradient noisy, and this can make the model take a "zig-zag" path to the optimal solution

## Training efficiency
- Use accelerate
  - ```accelerate launch --mixed_precision $DTYPE --num_machines $NUM_NODES --num_processes $NUM_GPUS --dynamo_backend 'no' finetune.py```
- We are using bf16 so i think here the mixed precision is not a problem
- Use Flash attention 2
- Always add this line export `export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` so we can see all the gpus for multi gpu training and not just one (in the end probably this isn't possible becasuse the gpus are assigned a memory size by default for each student I think and i get Out of Memory errors because of this)

## Heuristics of how things work
- First warmup ratio, at the beggining it is possible teh gradient norm will be zero becasue it will be scaled by a very small learning rate, but the computation are not wasted as Optimizers (e.g., Adam) still accumulate gradient statistics (mean/variance) during warmup, which are critical for later steps. But we won't be training on that part of the dataset, we are just computing the momentums for the optimizer.
- With smaller models you can use bigger learning rates, as we have fewer parameters the gradients have less averaging across parameters, so they are deterinistic
- For fine-tuning (SFT), the warmup ratio can depend on how different the downstream task is from the pretraining task.
  - If the task is very different, a higher warmup ratio might help the model adapt gradually.
  - If the task is similar, less warmup may be needed.


## Instruction finetuning datasets
- Probably the AYA dataset is not a good one as it is a very big multilingual dataset
- The FlanV2 dataset is a very high quality instruction finetuning dataset, so in reality with this dataset we should need to do to much the idea fof applying random templates
- The good instruciton finetuning datasets are:
  - FlanV2
  - Scirif data
  - the tulu if persona data (ifData)
  - noRobots
  - maybe a little bit of Oasst1, but it seems this dataset is biased to short user instructions and larger assistant answer than the responses


## Extra
- Talks about the problem of normalizing gradient accumulation https://unsloth.ai/blog/gradient


## Important
- It seems it is necessary to use the datacollator as it adds the special tokens correctly to the tokenization of the inputs, (I think the trainer at least does the label shifting)
  - It seems the way i do the tokenization is wrong compared to the one that sfttrainer does or unsloth does (and the sft one is also faster when training i don't know what is the difference). but pretty sure what im doing maybe is wrong (or maybe the sft trainer doesn't already add manually teh label shifting?), but the datacollator is not necessary it is jsut it seems the way im tokenizing is wrong as the loss that i get with my manual tokenization is bigger (2 instead of 1) and the grad norm at the beggining is 150 compared to sft with 2 or 4 (so pretty sure something is wrong as a big grad norm means the task is very different to what the model was trained for)

## Understanding SFT
- Do we train only on the completion or in the whole input
	- Our TA said that there shouldn't be any difference, but that aa friend of his working on a company says that if the output is always of longer length than the prompt the model can have a big loss (maybe because the model losses track of the prompt)
	- But i feel researching supposedly it should be only on the completion, but I also don't know what SFT trainer does by default
		- It seems SFT only applies completion loss only when using prompt and answer datasets but with the chat datsets it seems it doesn't


## Understanding the evaluation:

1. Token-Level Processing (Simplified)
Your prompt is tokenized into something like this (actual tokens depend on the tokenizer):

["Question:", " What", " is", " 2", "+", "2", "?", " Choices:", "\nA", ")", " 3", "\nB", ")", " 4", ...]
When evaluating choice A (A) 3):

The model sees the full prompt up to \nA.

It predicts the next token (which should be ")").

Then predicts the token after that (which should be " 3").

The key: It computes log probabilities for each required token in the choice sequence.

2. Scoring Choice A (A) 3)
Here’s how the model "scores" this choice internally:

Prompt Context:

"Question: What is 2+2?\nChoices:\nA"
Model generates logits for the next token (should be ")").

Logprob of ")" = -0.2 (example).

Next Token:

"Question: ... Choices:\nA)"
Now predicts the next token after ")" (should be " 3").

Logprob of " 3" = -1.0.

Total Logprob for Choice A:

logprob_A = logprob(")") + logprob(" 3") = -0.2 + (-1.0) = -1.2
3. Comparing All Choices
Repeat this for all options:

Choice	Tokens to Predict	Example Logprobs	Total Logprob
A) 3	")", " 3"	-0.2 + -1.0	-1.2
B) 4	")", " 4"	-0.2 + -0.1	-0.3
C) 5	")", " 5"	-0.2 + -1.9	-2.1
D) 6	")", " 6"	-0.2 + -1.6	-1.8
Winner: B) 4 (highest total logprob = -0.3).

4. Why This Works
No Generation: The model never outputs ") 3"—it just scores how likely those tokens are to follow the prompt.

Token-by-Token: Logprobs are additive across the sequence of tokens in each choice.

Answer Selection: The choice with the highest sum of logprobs wins.

5. Key Clarifications
Q: Does the model "see" other choices while scoring one?
No. Each choice (A) 3, B) 4, etc.) is scored independently as a continuation of the same base prompt.

Q: What if the choice is longer (e.g., "A) The number four")?
The model would score all tokens in the continuation:
")", " The", " number", " four" → sum their logprobs.

Normalization (_norm_nospace) divides by token count to avoid length bias.

Q: How is this different from generation?
Generation: Model freely produces tokens (e.g., "The answer is B)").

Logprob Scoring: Model silently evaluates how likely predefined tokens (like " 4") are to appear next.

6. Practical Implications
Efficiency: Faster than text generation (no decoding loop).

Accuracy: More reliable than parsing generated text (avoids formatting issues).

Reproducibility: Deterministic if the model and prompt are fixed.

Example Code (Pseudocode)
python
base_prompt = "Question: What is 2+2?\nChoices:\n"
choices = ["A) 3", "B) 4", "C) 5", "D) 6"]

scores = []
for choice in choices:
    input_text = base_prompt + choice
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Get logprobs for each token in the choice
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    # Calculate logprob for each token in the choice (simplified)
    choice_logprob = 0
    for i in range(len(input_ids[0]) - 1):
        token_logprob = logits[0, i, input_ids[0, i+1]].item()  # Logprob of next token
        choice_logprob += token_logprob
    
    scores.append(choice_logprob)

best_choice = choices[torch.argmax(torch.tensor(scores))]  # "B) 4"
Summary
The model scores each choice token-by-token as a continuation of the prompt.

It sums logprobs for all tokens in a choice (e.g., ")" + " 4" for B) 4).

The highest sum wins—no text generation needed.

This is why generation_size=-1 is optimal for multiple-choice QA! Let me know if you'd like to dive deeper into tokenization or normalization. 🚀