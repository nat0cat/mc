import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import os

# class for model information and usage
class ModelHandler:

    # initialize class with data
    def __init__(self):
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set cache
        user = os.getenv("USER")
        pth = f"~/scratch/{user}/hf_cache"
        self.cache_dir = os.path.expanduser(pth)

        # model information
        with open("models.json") as f:
            self.model_info = json.load(f)
        self.model = None
        self.m_name = ""

        # map to determine class
        self.qwen_map = {
            "2": Qwen2VLForConditionalGeneration,
            "2.5": Qwen2_5_VLForConditionalGeneration
        }

    # load vlm
    def load_vlm(self, m, v):
        print("... loading vlm")

        # find model path to load
        if v !="": path = self.model_info[m]["version"][v]
        else: path = self.model_info[m]["path"]
        self.m_name = path
        print("... ... model: ", path)

        # get the model class using the map
        cls = self.qwen_map[self.model_info[m]["cls"]]

        # load model
        vlm = cls.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir)
        processor = AutoProcessor.from_pretrained(path, use_fast=True, cache_dir=self.cache_dir)
        self.model = (vlm, processor)

        # return vlm and processor
        return self.model

    # pass vlm to caller
    def get_vlm(self, m=None, v=None):
        if self.model is None:
            self.load_vlm(m, v)
        return self.model

    # pass model path to caller
    def get_vlm_name(self):
        return self.m_name


# run inf with vlm
def run_vlm(model, processor, message, config):
    # format input
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image, _ = process_vision_info(message)
    inputs = processor(text=text, images=image, padding=True, return_tensors="pt").to(model.device)

    # run inference
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **config
        )
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # format output
    output = response.split("assistant")[-1].strip()
    return output





