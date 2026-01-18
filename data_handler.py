import os
from datasets import load_dataset

# class to manage datasets and data
class DataHandler(object):
    def __init__(self):
        # load dataset
        data_pth = os.path.expanduser(f"~/projects/aip-btaati/shared/armen/miccai2026/vlaa/data.jsonl")
        self.full_dataset = load_dataset("json", data_files=data_pth)["train"]

        # dataset specifics
        self.dataset = None  # specific subset
        self.dt_name = ""    # name of the subset


    # load dataset
    def load_dt(self, dt_name):
        self.dt_name = dt_name
        self.dataset = self.full_dataset.filter(lambda item: item["dataset_name"] == dt_name)


    # returns loaded dataset
    def get_data(self):
        # if the subset has not been specified, then return full dataset
        if self.dataset is None: return self.full_dataset

        # returns the specific subset
        return self.dataset


    # return name of dataset
    def get_dataset_name(self):
        return self.dt_name



# returns formatted message as input for qwen inference
def get_message(item):
    # content component of message
    content = []

    # format images in content
    for p in item["image_paths"]:
        content.append({
            "type": "image",
            "image": os.path.expanduser(f"~/projects/aip-btaati/shared/armen/miccai2026/{p}")
        })

    # format prompt
    prompt = get_prompt(item)

    # build message
    content.append({"type": "text", "text": prompt})
    message = [{"role": "user", "content": content}]

    # return message
    return message


# returns formatted prompt
def get_prompt(item):
    # format question using item questions and options
    question = f"{item['question']} \n{item['options_dict']}"

    # template for prompt
    desc = "You are given an image and corresponding multiple choice question:"
    task = "Enclose your reasoning in these tags <think> </think>, and state your final answer as a multiple choice letter <answer> </answer>"

    # build and return prompt
    prompt = f"{desc} \n{question} \n{task}"
    return prompt


# finds a unique filename
def get_filename(path):
    # split into name and extension
    base, ext = os.path.splitext(path)

    # loop vars
    c = 1
    n_pth = path

    # iterate until new file name is found
    while os.path.exists(n_pth):
        n_pth = f"{base}_{c}{ext}"
        c += 1

    # return filename
    return n_pth