# Inference Code Usage
### Check: ```run_sanity_check.sh```
This script runs a very short inference job to test if the pipeline is working, and if the environment is correctly set up.  
It runs through 2 rows from the dataset, only once: ```max_examples=2, k=1```  

Before running the script, modify the variable ```VENV_PTH``` to a specific virtual environment.  

Usage: 
```
sbatch run_sanity_check.sh
```



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

### ```inference_utils.py```
This module stores two functions that are used for running this pipeline.
1. ```eval_row(message, vlm, tokenizer, config, k)```:
   - Given: a formatted input message associated with one dataset item, a vlm, inference configurations and k iterations
   - The function calls ```model_handler.run_vlm()``` repeatedly, to construct a list of ```k``` responses
2. ```run_inference(md_handler, dt_handler, config, k, max_examples)```:
   - Given: handler objects, inference configurations, k, and the max number of examples
   - The function iterates over ```max_examples``` in the dataset, and calls ```eval_row(k)``` to run inference ```k``` times
   - For each row in the dataset, it constructs a dictionary called ```entry``` that contains row information, and the list of ```k``` responses
     ```
     entry = {
         "id": item["dataset_index"],
         "images": item["image_paths"],
         "question": item["question"],
         "choices": item['options_dict'],
         "full_prompt": dt.get_prompt(item),
         "answer_value": item["answer"],
         "answer_label": item["answer_label"],
         "responses": responses,
         "dataset_name": item["dataset_name"]
     }
     ```
  - This constructs a list of entries that are returned to the caller.


### ```inference.py```
This module consists of only a main function, that uses the other modules to run inference.  
When running this script, the following arguments can be used,
- model: 
  - the name of the model to run inference on
  - options: qwen2.5, qwen2, medr1, medvlmr1, medvlthinker
  - these options are entries in the ```models.json``` file
- version:
  - the specific version of the model to run
  - leave it as "" for the default value specified in ```models.json[model_name][path]```
  - options can be found in ```models.json[model_name][version]```
- d_name:
  - name of the specific dataset
  - options: MMMU-medical, slake_closed, vqa_rad_closed, pmc_vqa, pathvqa_closed, MedXpertQA-MM
- max_examples: maximum number of entries in the dataset to run inference on
- k: number of iterations on the same entry in the dataset
- temp: HuggingFace ```model.generate``` config parameter, default=0.7
- top_p: HuggingFace ```model.generate``` config parameter, defuult=0.9
- top_k: HuggingFace ```model.generate``` config parameter, default=0
- max_tok: HuggingFace ```model.generate``` config parameter, default=2048

The results are saved to: ```inf_results/{model_name}-{dataset_name}-k{k}-max{max_examples}.jsonl```

