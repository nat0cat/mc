import model_handler as md
import data_handler as dt

# runs inference k times over a row
def eval_row(message, vlm, tokenizer, config, k):
    # list of responses from running inference
    responses = []

    # repeat k times
    for i in range(k):
        output = md.run_vlm(vlm, tokenizer, message, config)  # run inference
        responses.append(output)                              # add to list

    # return list of k responses
    return responses


# runs inference over max_examples in the dataset
def run_inference(md_handler, dt_handler, config, k, max_examples):
    # get loaded values
    vlm, tokenizer = md_handler.get_vlm()  # vlm
    dataset = dt_handler.get_data()        # dataset

    # inference data to save to file
    inference_data = []

    # iterate over dataset
    for i, item in enumerate(dataset):
        # break at max examples
        if i == max_examples: break

        # get formatted message
        message = dt.get_message(item)

        # run inference k times
        responses = eval_row(message, vlm, tokenizer, config, k)

        # construct entry for dataset item
        entry = {"id": item["dataset_index"],
                 "images": item["image_paths"],
                 "question": item["question"],
                 "choices": item['options_dict'],
                 "full_prompt": dt.get_prompt(item),
                 "answer_value": item["answer"],
                 "answer_label": item["answer_label"],
                 "responses": responses,
                 "dataset_name": item["dataset_name"]}
        inference_data.append(entry)

        # display every 10 iterations
        if (i + 1) % 10 == 0: print(f"{i+1}/{len(dataset)}")

    # return the list of entries
    return inference_data


