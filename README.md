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
  - ```
  accelerate launch
 --mixed_precision $DTYPE
 --num_machines $NUM_NODES
 --num_processes $NUM_GPUS
--dynamo_backend 'no'
 finetune.py
 ```
- We are using bf16 so i think here the mixed precision is not a problem
- Use Flash attention 2
- Always add this line export `export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` so we can see all the gpus for multi gpu training and not just one (in the end probably this isn't possible becasuse the gpus are assigned a memory size by default for each student I think and i get Out of Memory errors because of this)

## Heuristics of how things work
- First warmup ratio, at the beggining it is possible teh gradient norm will be zero becasue it will be scaled by a very small learning rate, but the computation are not wasted as Optimizers (e.g., Adam) still accumulate gradient statistics (mean/variance) during warmup, which are critical for later steps. But we won't be training on that part of the dataset, we are just computing the momentums for the optimizer.
- With smaller models you can use bigger learning rates, as we have fewer parameters the gradients have less averaging acrros parameters, so they aere deterinistic


## Extra
- Talks about the problem of normalizing gradient accumulation https://unsloth.ai/blog/gradient
