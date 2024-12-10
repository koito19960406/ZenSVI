def infer_type(x):
    """Infer the type of a given input.

    This function attempts to convert a string input into an integer or a float.
    If the input is not a string or cannot be converted, it returns the input as is.

    Args:
        x: The input value to infer the type from. Can be a string, int, float, or any other type.

    Returns:
        The input value converted to int, float, or the original value if conversion is not possible.
    """
    if not isinstance(x, str):
        return x

    try:
        x = int(x)
        return x
    except ValueError:
        pass

    try:
        x = float(x)
        return x
    except ValueError:
        pass

    return x


def parse_unknown(unknown_args):
    """Parse unknown arguments into a dictionary.

    This function takes a list of unknown arguments in the form of strings,
    splits them into key-value pairs, and returns a dictionary with the keys
    stripped of leading dashes and values converted to their inferred types.

    Args:
        unknown_args: A list of strings representing unknown arguments,
                      where each argument can be in the form 'key=value' or just 'key'.

    Returns:
        A dictionary mapping keys to their inferred values.
    """
    clean = []
    for a in unknown_args:
        if "=" in a:
            k, v = a.split("=")
            clean.extend([k, v])
        else:
            clean.append(a)

    keys = clean[::2]
    values = clean[1::2]
    return {k.replace("--", ""): infer_type(v) for k, v in zip(keys, values)}
