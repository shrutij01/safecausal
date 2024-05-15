from datasets import load_dataset

from data import ana



def load_dataset(dataset_name, **kwargs):
    if dataset_name == "ana":
        return ana.generate_data(**kwargs)
    else:
        if dataset_name == "truthful_qa":
            dataset_params = {
                'split': 'validation',
                'name': 'multiple_choice',     
            }
        dataset = load_dataset(dataset_name, **dataset_params)    


def append_instruction(contexts, instruction):
    instruction_plus_contexts = []
    for context in contexts:
        instruction_plus_contexts.append([str(instruction) + ' ' + str(context)])
    return instruction_plus_contexts

