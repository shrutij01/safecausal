from datasets import load_dataset

from data import ana



def load_dataset(dataset_name, **kwargs):
    if dataset_name == "ana":
        data = ana.generate_data(**kwargs)
        instruction = "Swap the letters in this string."
        cfc1_tuples, cfc2_tuples = data[:, 0], data[:, 1]
    else:
        if dataset_name == "truthful_qa":
            dataset_params = {
                'split': 'validation',
                'name': 'multiple_choice',     
            }
        data = load_dataset(dataset_name, **dataset_params)
        instruction = "Label as 0 for False and 1 for True."
    return cfc1_tuples, cfc2_tuples, instruction    


def append_instruction(contexts, instruction):
    instruction_plus_contexts = []
    for context in contexts:
        instruction_plus_contexts.append([str(instruction) + ' ' + str(context)])
    return instruction_plus_contexts

