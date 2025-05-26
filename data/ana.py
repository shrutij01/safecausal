import numpy as np


def swap_pairs(n, d_type):
    """
    Generates an array of n pairs where each pair consists of two elements, then swaps the elements to create the second item in each pair 
    Returns: array where each row is of the format [AB, BA] or [42, 24]
    """
    
    result = None
    if d_type == "str":
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
        swapped_tuples = np.core.defchararray.add(
            second_letters, first_letters
        )
        result = np.stack((tuples, swapped_tuples), axis=1)
    elif d_type == "int":
        numbers = np.random.randint(10, 100, size=n)

        def swap_digits(num):
            swapped = int(f"{num % 10}{num // 10}")
            return f"{swapped:02d}"

        swapped_numbers = np.vectorize(swap_digits)(numbers)
        result = np.vstack((numbers, swapped_numbers)).T
    return result


def cycle_strings(n, string_length, cycle_distance, d_type):
    """
    Generates list of n tuples each containing a random string/number and a permutation of it
    """
    
    tuples = None
    if d_type == "str":
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
    elif d_type == "int":
        lower_limit = 10 ** (string_length - 1)
        upper_limit = 10**string_length - 1

        # Generate n random numbers within the specified digit range
        numbers = np.random.randint(lower_limit, upper_limit + 1, size=n)

        # Function to cycle digits and correctly maintain leading zeros
        def cycle_digits(number):
            num_str = f"{number:0{string_length}d}"  # Format number with leading zeros based on digit length
            cycled = (
                num_str[cycle_distance:] + num_str[:cycle_distance]
            )  # Correctly cycle the digits
            return cycled

        # Convert and cycle numbers
        cycled_numbers = np.array([cycle_digits(num) for num in numbers])

        # Combine original and cycled numbers into pairs, with original numbers formatted
        formatted_numbers = np.array(
            [f"{num:0{string_length}d}" for num in numbers]
        )
        tuples = np.vstack((formatted_numbers, cycled_numbers)).T

    return np.asarray(tuples)


def generate_data(
    task_type, n, d_type, string_length=None, cycle_distance=None
):
    if task_type == "swap":
        return swap_pairs(n=n, d_type=d_type)
    elif task_type == "cycle":
        if string_length is None or cycle_distance is None:
            raise ValueError(
                "For cycling, string_length and cycle_distance must be provided"
            )
        return cycle_strings(
            n=n,
            d_type=d_type,
            string_length=string_length,
            cycle_distance=cycle_distance,
        )
    else:
        raise ValueError(
            "Unsupported task specified. Choose 'swap' or 'cycle'."
        )
