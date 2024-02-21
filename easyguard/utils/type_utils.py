from functools import wraps
from inspect import signature


def typecheck(*ty_args, **ty_kwargs):
    """Decorator, used to check each argument for a function or class
    example:
        @typecheck(int, name=str)
        def person(age, name):
            age += 1
            return f"age: {age}, name: {name}"
        >>> person(31, "jack")
        age: 32, name: jack
        >>> person(31.5, "jack")
        TypeError: Argument `age` must be <class 'int'>
    """

    def decorate(func):
        # If in optimized mode, disable type checking
        if not __debug__:
            return func

        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError("Argument `{}` must be {}".format(name, bound_types[name]))
            return func(*args, **kwargs)

        return wrapper

    return decorate
