import model_handler as md
import data_handler as dt
import inference_utils as iu
import argparse
import json
import os

def main():
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="medvlthinker")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--d_name", type=str, default="mmmu")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_tok", type=int, default=2048)
    args = parser.parse_args()

    # set inference parameters
    model_name = args.model
    ver = args.version
    dataset_name = args.d_name
    model_config = dict(
        do_sample=True,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_tok
    )
    max_examples = args.max_examples
    k = args.k

    # outfile name
    os.makedirs("inf_results", exist_ok=True)
    basename = f"inf_results/{model_name}-{dataset_name}-k{k}-max{max_examples}.json"
    outfile = dt.get_filename(basename)

    # load the handler objects
    print("... loading objects")
    md_handler = md.ModelHandler()
    md_handler.load_vlm(model_name, ver)
    dt_handler = dt.DataHandler()
    dt_handler.load_dt(dataset_name)

    # display
    print("... running inference with the following parameters:")
    print(f"\tmodel: {md_handler.get_vlm_name()}")
    print(f"\tdataset name: {dataset_name}")
    print(f"\tmodel config: {model_config}")
    print(f"\tmax examples: {max_examples}")
    print(f"\tk: {k}")
    print("... results will be saved to: ", outfile)
    print("=========================================================")

    # run inference
    inference_data = iu.run_inference(md_handler, dt_handler, model_config, k, max_examples=max_examples)

    # save to json
    with open(outfile, "w") as f:
        json.dump(inference_data, f)

    # display
    print("=========================================================")
    print("results saved to: ", outfile)

    return 0


if __name__ == "__main__":
    main()