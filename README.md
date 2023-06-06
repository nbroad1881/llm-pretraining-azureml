# llm-pretraining-azureml


Currently has 3 size options for running pre-training for a GPTBigCode Model (like starcoder)
- 1B
- 2B
- 5B


You will need to 
1. add your Azure ML config.json file to this directory
2. point the script to the correct text files
3. point the script to a trained tokenizer (should include a config.json too that has the right vocab size)
4. modify compute instances, hyperparameters to suit your needs
5. modify the deepspeed_config.json depending on what you want to offload


[run_pipeline.ipynb](./run_pipeline.ipynb) has everything to run the training.


To do  
- [ ] Add flash attention
- [ ] bf16 deepspeed config
- [ ] torch 2.0 with compile

