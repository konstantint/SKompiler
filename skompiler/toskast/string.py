"""
String to SKAST translator.
"""
import ast
from .python import translate as from_python


def translate(expr):
    """Convert a given (restricted) Python code string to a SK-AST.
       
       So far we only need this functionality for debugging purposes,
       hence instead of implementing a full-fledged parser, we rely on
       Python's ast.parse.

       This means that if the expression contains non-Python code
       or Python code which we cannot translate, you will get cryptic errors.

       >>> expr = translate("12.4 * (X1[25.3] + Y)")
       >>> print(str(expr))
       (12.4 * (X1[25.3] + Y))
       >>> expr = translate("a=X; b=a+2; 12.4 * (a[25.3] + b + y)")
       Traceback (most recent call last):
       ...
       ValueError: Subscripting named references is not supported
       >>> expr = translate("a=X; b=a+2; 12.4 * (a + b + y)")
       >>> print(str(expr))
       {
       $a = X;
       $b = ($a + 2);
       (12.4 * (($a + $b) + y))
       }
       """
    return from_python(ast.parse(expr))
