# Inference Code Usage

## Model Information
The code supports the following models:
- qwen2VL (any size)
- qwen2.5VL (any size)
- MedVLThinker versions (3B or 7B):
  - RL_m23k
  - RL_PMC
  - SFT_m23k_RL_PMC
  - RL_m23k_RL_PMC
- Med-R1
- MedVLM-R1

The file ```models.json``` stores the HuggingFace model paths, the supported versions as listed above, and their associated Qwen class for instantiation (either Qwen2VL or Qwen2.5VL).  
To include support for additional models, the file can be updated to include their information.

## "Handler" Modules
The "handler" modules, consist of an object to manage information, and functions that perform related operations. 
There are two such classes:
1. ```data_handler.py```
2. ```model_handler.py```

### Dataset Management Usage: ```data_handler.py```
The dataset is loaded from: ```~/projects/aip-btaati/shared/armen/miccai2026/vlaa/data.jsonl```  
- ```load_dt(item_name)```:
  - loads a subset from the full dataset
  - for example, to load specifically the "MMMU-medical" dataset, the usage would be: ```load_dt("MMMU-medical")```
  - subsets in the dataset: MMMU-medical, slake_closed, vqa_rad_closed, pmc_vqa, pathvqa_closed, MedXpertQA-MM
- ```get_data()```:
  - returns the dataset loaded
  - if a subset was not already loaded, it returns the full dataset
 - ```get_dataset_name()```:
   - returns the name of the dataset loaded
   - specifically one of the following: MMMU-medical, slake_closed, vqa_rad_closed, pmc_vqa, pathvqa_closed, MedXpertQA-MM
- ```get_message(item)```:
  a static function that returns formatted input for qwen inference
    ```
    message = [{
      "role": "user",
      "content": [
        {"type": "image", "image": <image_path>}
        {"type": "text", "text": <prompt>}]
    }]
    ```
- ```get_prompt(item)```:
  a static function that returns a prompt to be passed into a qwen-vl model
  ```
  You are given an image and corresponding multiple choice question:
  {item["question"]}
  {item["options_dict]}
  Enclose your reasoning in these tags <think> </think>, and state your final answer as a multiple choice letter <answer> </answer>
  ```
- ```get_filename(path)```:
  a static function that searches for a duplicate filename, and adds a number at the end

### Model Management Usage: ```model_handler.py```
Loads model to a cache: ```~/scratch/{$USER}/hf_cache```  
Using the ```models.json``` file, it loads the specified model and version, using the associated qwen class for instantiation.  
Returns model information to user:
- ```get_vlm(m, v)```:
  returns the vlm and tokenizer
- ```get_vlm_name()```:
  returns the HuggingFace path of the model

Contains a function to run inference:
```run_vlm(model, processor, message, config)```:
- formats message using :
  ```processor.apply_chat_template``` and ```process_vision_info```
- formats input using: ```processor```
- calls ```model.generate()``` to produce an output using formatted input and specified configs
- output is converted to a readable response using: ```processor.batch_decode```

## Inference Modules
There are two files:
1. ```inference_utils.py```
2. ```inference.py```

Where ```inference_utils.py``` contains two functions, ```eval_row()```, and ```run_inference```.
These functions are used by the main function in ```inference.py```.
The results are saved to a directory ```inf_results``` as: ```inf_results/{model_name}-{dataset_name}-k{k}-max{max_examples}.jsonl```

