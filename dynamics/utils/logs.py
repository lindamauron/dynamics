import numpy as np
import json
import warnings


def save_log(filename: str, log: dict):
    r"""
    Saves the log as a filename.log file
    Args:
        filename : The name of the file to save the log in.
        log : The dictionary to save.
    """
    import orjson

    with open(filename + ".log", "wb") as io:
        io.write(
            orjson.dumps(
                log,
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )

    return


def append_dict(data1, data2):
    """
    Append values from two dictionaries recursively.
    Args:
        data1 (dict): The first dictionary containing initial values.
        data2 (dict): The second dictionary containing values to append.
    Returns:
        dict: A new dictionary with values from `data2` appended to the corresponding values in `data1`.

    Example:
        >>> data1 = {'a': [1, 2], 'b': {'c': [3, 4]}}
        >>> data2 = {'a': [5, 6], 'b': {'c': [7, 8]}}
        >>> append_dict(data1, data2)
            {'a': [1, 2, 5, 6], 'b': {'c': [3, 4, 7, 8]}}
    """

    def iterate_dict(d1, d2):
        new_dict = {}
        for k in d1.keys():

            new_dict[k] = {}
            if k not in d2.keys():
                new_dict[k] = d1[k]
                continue
            else:
                if isinstance(d1[k], dict):
                    new_dict[k] = iterate_dict(d1[k], d2[k])
                else:
                    new_dict[k] = np.append(d1[k], d2[k]).tolist()

        return new_dict

    new_dict = iterate_dict(data1, data2)
    for k in data2.keys():
        if k not in new_dict.keys():
            new_dict[k] = data2[k]

    return new_dict


def join_logs(*args, filename=None):
    """
    joins multiple logs together and discards the doubled parts
    Args:
        *args: multiple logs to join, in order
        filename: name of the file into which the resulting log should be saved
            if not provided, the log is not saved

    Returns:
        new_log without repetition in the values
    """
    logs = list(args)
    n = len(logs)
    log1 = logs[0]

    for i in range(1, n):
        log2 = logs[i]

        ## append two logs
        new_log = append_dict(log1, log2)

        ## delete double entries
        for k in new_log.keys():
            mask = np.ones(len(new_log[k]["iters"]), dtype=bool)
            try:
                start = np.where(np.isclose(log1[k]["iters"], log2[k]["iters"][0]))[0][
                    0
                ]
                end = len(log1[k]["iters"])
                if end == start:
                    start += 1
                    end += 2
                mask[start:end] = 0

                # print(k, start, new_log[k]['iters'][start-2:end+2])
            except IndexError:
                warnings.warn(
                    f"{k} seems to have no repetition in its indices, nothing will be deleted"
                )
                pass

            for key in new_log[k]:
                if key == "Mean":
                    for i in ["real", "imag"]:
                        new_log[k][key][i] = list(np.array(new_log[k][key][i])[mask])

                else:
                    new_log[k][key] = list(np.array(new_log[k][key])[mask])

        log1 = new_log

    ## Verify that all entries have the same length
    length = len(new_log["t"]["iters"])
    print(length)
    for k in new_log.keys():
        for key in new_log[k].keys():
            if key == "Mean":
                for i in new_log[k][key].keys():
                    if (k != "Generator" and len(new_log[k][key][i]) != length) or (
                        k == "Generator" and len(new_log[k][key][i]) != length - 1
                    ):
                        warnings.warn(
                            f"The length of [{k}][{key}] should be {length} but is {len(new_log[k][key][i])}."
                        )
            else:
                if (k != "Generator" and len(new_log[k][key]) != length) or (
                    k == "Generator" and len(new_log[k][key]) != length - 1
                ):
                    warnings.warn(
                        f"The length of [{k}][{key}] should be {length} but is {len(new_log[k][key])}."
                    )
                    if k == "acc" and len(new_log[k][key]) == length - 1:
                        new_log[k]["iters"].insert(0, 0)
                        new_log[k]["value"].insert(0, new_log[k]["value"][0])

    if filename is not None:
        save_log(filename, new_log)
    return new_log
