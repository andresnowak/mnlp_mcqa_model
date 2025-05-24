
- [x] ⁠remove NLP4education form dataset
- [x] ⁠⁠Maybe see about increasing a little bit the dataset of MCQA
- [ ] ⁠Test more with direct single letter MCQA and try directly from the finetuned model
- [ ] ⁠Try something again with instruction finetuend one (without the NLP and maybe without other languages, but i don’t think that part affected it). like i don’t know how many examples where truncated and the eos token was visible because of this, i have to also fix that
- [ ] ⁠⁠We at least understand the evaluation scheme a lot better, the prompt used and they just use the letter
- [ ] ⁠See about how to add metadata for each model commit to see with which parameters it was trained, and to also say that the first one also had the NLP4education dataset
- [ ] Maybe it is a problem, so fix the eos token being added at the end of the complete prompt + answer, because the truncation happens after and maybe the eos token won't appear (and maybe this is a problem)
- [ ] Maybe change the training scheme for MCQA to train like if it was text generation, so basically it finishes with just a letter the answer and eos token, because the way we are doing it right now, the model will generate until max tokens pretty sure