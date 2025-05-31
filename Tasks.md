
- [x] ⁠remove NLP4education form dataset
- [x] ⁠⁠Maybe see about increasing a little bit the dataset of MCQA
- [ ] ⁠Test more with direct single letter MCQA and try directly from the finetuned model
  - [x] Right now the training seems to be wrong, like th softmax is applied differently when we have different number of options (the denominator size is different) and i think this is wrong like you can be doing different types of losses with different amount of classes the calculations are different even though it isi being normalized (it is easier to increase the probability when you have less options)
    - We fix this by just training on examples with 4 choices
- [ ] ⁠Try something again with instruction finetuned one (without the NLP and maybe without other languages, but i don’t think that part affected it). like i don’t know how many examples where truncated and the eos token was visible because of this, i have to also fix that
  - [x] Try experiment with random templates for instruction finetuning
    - It worked well, but right now my instruction finetuning seems to remove reasoning (lets thinkg step by step) abilities
- [x] ⁠See about how to add metadata for each model commit to see with which parameters it was trained, and to also say that the first one also had the NLP4education dataset
- [x] Maybe it is a problem, so fix the eos token being added at the end of the complete prompt + answer, because the truncation happens after and maybe the eos token won't appear (and maybe this is a problem)
  - Don't know if this a problem in the end, at least already did the change where i remove all things that have more than 15_000 characters
- [x] Maybe change the training scheme for MCQA to train like if it was text generation, so basically it finishes with just a letter the answer and eos token, because the way we are doing it right now, the model will generate until max tokens pretty sure (**Working on this**)
  - So the error here will only be done on the choice and answer, like we put in the prompt that it only has to ouput one letter and we only do the loss on letter but for all of them like the answer will be the letter and text of the option
  - It seems this method also works well, but it is not like gives more extra abilities
- [x] Use unsloth trainer instead in finetune.py to fix the error of the gradient accumulation
- [x] Fix problem with Xformers
- [x] Try and use a higher warmup ratio for the instruction finetuning
- [ ] Try and use the NEFTune method

⁠⁠We at least understand the evaluation scheme a lot better, the prompt used and they just use the letter for the likelihood it seems