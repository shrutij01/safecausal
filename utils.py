import ana


def load_toy_dataset(dataset_name, **kwargs):
    if dataset_name == "ana":
        return ana.generate_data(**kwargs)
