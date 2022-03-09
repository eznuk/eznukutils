def get_opt_kwargs(to_extract, **kwargs):
    """Extracts variables from a dictionary to_extract, 
    if they are present. Otherwise, apply standard values.
    
    The variables to extract are given in **kwargs, together
    with their standard values.
    
    Returns a dict_values list. Access it in the form of
    a, b, c = get_opt_kwargs(to_extract, a=1, b=2, c=3)
    For a single value:
    a, = get_opt_kwargs(to_extract, a=1)
    """
    
    for var in kwargs:
        if var in to_extract:
            kwargs[var] = to_extract[var]
    return kwargs.values()

def make_iter(x):
    """If x is no iterable, return a list containing itself, 
    which is iterable.
    Ff x is an iterable, return x.
    Ff it is a string (which is iterable), still return [x]
    to avoid iteration over single letters.
    """
    if isinstance(x, str):
        x = [x]
    else:
        try: _ = iter(x)
        except: x = [x]
        
    return x
