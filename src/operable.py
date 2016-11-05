"""Module that provides functionality for operations with functions."""

import numbers
import operator

class Operable:
    """Decorator that allows to apply standard (as add, multiply etc.) operators to functions."""

    def __init__(self, f):
        """Initialize instance with function or number."""
        self.function = (lambda *args: f) if isinstance(f, numbers.Number) else f

    def __call__(self, *args):
        """Invoke with passed arguments."""
        return self.function(*args)


def _op_to_function_op(oper):
    """Convert operator to operator on functions."""
    def function_op(self, operand):
        """Converted operator."""
        def func(*args):
            """Result function."""
            return oper(self(*args), Operable(operand)(*args))
        return Operable(func)
    return function_op


# to allow int + operable for e. g.
RIGHT_OPERATORS = {
    '__add__': '__radd__',
    '__sub__': '__rsub__',
    '__mul__': '__rmul__',
    '__div__': '__rdiv__',
    '__truediv__': '__rtruediv__',
    '__floordiv__': '__rfloordiv__',
    '__mod__': '__rmod__',
    '__divmod__': '__rdivmod__',
    '__pow__': '__rpow__',
    '__lshift__': '__rlshift__',
    '__rshift__': '__rrshift__',
    '__and__': '__rand__',
    '__xor__': '__rxor__',
    '__or__': '__ror__'
}

for name, op in [(name, getattr(operator, name)) for name in dir(operator) if "__" in name]:
    try:
        op(1, 2)
    except TypeError:
        pass
    else:
        setattr(Operable, name, _op_to_function_op(op))
        if name in RIGHT_OPERATORS:
            setattr(Operable, RIGHT_OPERATORS[name], _op_to_function_op(op))
