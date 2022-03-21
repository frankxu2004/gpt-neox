import json
import multiprocessing
import os
import re

from datasets import load_dataset, load_metric
from tqdm import tqdm


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()


def main():
    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    num_workers = multiprocessing.cpu_count()

    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")
    code_eval_metric = load_metric("code_eval")
    
    num_gen_per_task = 100
    generated_all = []
    with open('0.4b-150k-0.8.output.jsonl') as generated_file:
        for line in generated_file:
            generated_all.append(json.loads(line.strip()))
    
    # Generate completions for evaluation set
    n_tasks = len(human_eval["test"])
    generations, references = [], []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = human_eval["test"][task]["prompt"].strip()

        for sample in generated_all[task*num_gen_per_task:(task+1)*num_gen_per_task]:
            assert sample['context'] == prompt
            task_generations.append(prompt + first_block(sample['text']))

        generations.append(task_generations)
        test_func = human_eval["test"][task]["test"]
        entry_point = f"check({human_eval['test'][task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)

    # Evaluate completions with "code_eval" metric
    pass_at_k, _ = code_eval_metric.compute(
        references=references, predictions=generations, num_workers=num_workers
    )
    print(f"Results: {pass_at_k}")


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
