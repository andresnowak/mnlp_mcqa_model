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