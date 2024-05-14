import numpy as np


import numpy as np


def swap_pairs(n):
    # 1. generate number arrays
    first_letters = np.random.randint(0, 26, n)
    second_letters = np.random.randint(0, 26, n)

    # 2. ensure distinct
    mask = first_letters == second_letters
    while np.any(mask):
        second_letters[mask] = np.random.randint(0, 26, mask.sum())
        mask = first_letters == second_letters

    # 3. convert number arrays to uppercase letters
    first_letters = np.char.mod("%c", first_letters + 65)
    second_letters = np.char.mod("%c", second_letters + 65)

    # 4. form the tuples (AB, BA)
    tuples = np.core.defchararray.add(first_letters, second_letters)
    swapped_tuples = np.core.defchararray.add(second_letters, first_letters)
    result = np.stack((tuples, swapped_tuples), axis=1)

    return result.tolist()


def cycle_strings(n, string_length, cycle_distance):
    # 0. some checks and balances
    if string_length > 26:
        return "String length cannot exceed 26, as there are only 26 distinct letters in the alphabet."
    if string_length <= 0 or n <= 0:
        return "Please provide positive values for number of tuples and string length."

    effective_cycle_distance = cycle_distance % string_length
    if cycle_distance != effective_cycle_distance:
        print(
            f"Warning: cycle_length of {cycle_distance} exceeds string length. "
            f"Using effective cycle_length of {effective_cycle_distance}."
        )

    # 1. generate n random tuples of distinct alphabets
    letters = np.array(
        [
            np.random.choice(26, size=string_length, replace=False)
            for _ in range(n)
        ]
    )

    # 2. convert numeric indices to letters
    letters = np.char.mod("%c", letters + 65)

    # 3. generate tuples with original and cycled strings
    original_strings = ["".join(l) for l in letters]
    cycled_strings = [
        "".join(np.roll(l, -effective_cycle_distance)) for l in letters
    ]
    tuples = list(zip(original_strings, cycled_strings))

    return tuples


def generate_data(task, n, string_length=None, cycle_distance=None):
    if task == "swap":
        return swap_pairs(n)
    elif task == "cycle":
        if string_length is None or cycle_distance is None:
            raise ValueError(
                "For cycling, string_length and cycle_distance must be provided"
            )
        return cycle_strings(n, string_length, cycle_distance)
    else:
        raise ValueError(
            "Unsupported task specified. Choose 'swap' or 'cycle'."
        )
