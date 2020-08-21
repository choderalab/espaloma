def get_reversals(data_dict):
    """ if data_dict has key tuples that look like (a,b,c)
    and list-of-tuple values that look like [(d, i), (e, i), (f, i), (g, i), ...]

    return a new dict where the key tuples are reversed, and the tuples in the value lists are reversed
    (c,b,a) : [(i, d), (i, e), (i, f), (i, g), ...]

    sometimes needed for message passing on heterograph
    """
    new_dict = dict()
    for (key, value) in data_dict:
        new_dict[key[::-1]] = [v[::-1] for v in value]
    return new_dict