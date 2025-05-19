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
  - So what we have to do for formatting is do {Question} \n {Answer}
- Use Flash attention and accelerate
- Maybe use QLora
- Does SFTT and in general when doing finetuning, is the loss done on Question and Answer? or just the answer
- Use packing maybe, because it helps putting multiple examples in a single prompt up to max seq length (packing adds eos sentence tokens between examples)


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