from datasets import ana



def load_toy_dataset(dataset_name, **kwargs):
    if dataset_name == "ana":
        return ana.generate_data(**kwargs)


def append_instruction(contexts, instruction):
    instruction_plus_contexts = []
    for context in contexts:
        instruction_plus_contexts.append([instruction, context, 0])
    return instruction_plus_contexts

