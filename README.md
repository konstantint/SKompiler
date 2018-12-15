SKompiler: Translate trained SKLearn models to executable code in other languages
================================================================================

The package provides a tool for transforming trained SKLearn models into other forms, such as SQL queries, Excel formulas or Sympy expressions (which, in turn, can be translated to code in a variety of languages, such as C, Javascript, Rust, Julia, etc).

Installation
------------

The simplest way to install the package is via `pip`:

    $ pip install SKompiler[full]


Note that the `[full]` option includes the installations of `sympy`, `sqlalchemy` and `astor`, which are necessary if you plan to convert `SKompiler`'s expressions to `sympy` expressions (which, in turn, can be compiled to many other languages) or to SQLAlchemy expressions (which can be further translated to different SQL dialects) or to Python source code. If you do not need this functionality (say, you only need the raw `SKompiler` expressions or perhaps only the SQL conversions without the `sympy` ones), you may avoid the forced installation of all optional dependencies by simply writing

    $ pip install SKompiler

(you are free to install any of the required extra dependencies, via separate calls to `pip install`, of course)

Usage
-----

### Introductory example

Let us start by walking through a simple example. We begin by training a model on a simple dataset, e.g.:

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_iris(True)
    m = RandomForestClassifier(n_estimators=3, max_depth=3).fit(X, y)

Suppose we need to express the logic of `m.predict` in SQLite. Here is how we can achieve that:

    from skompiler import skompile
    expr = skompile(m.predict)
    sql = expr.to('sqlalchemy/sqlite')

Voila, the value of the `sql` variable is a multi-step query of the form:

    WITH _tmp1 AS
    (SELECT .... FROM data)
    _tmp2 AS
    ( ... )
    SELECT ... from _tmp2 ...

It corresponds to the `m.predict` computation. Let us check how we can use it in a query.
Let us import the data into an in-memory SQLite database:

    import sqlalchemy as sa
    import pandas as pd
    conn = sa.create_engine('sqlite://').connect()
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4']).reset_index()
    df.to_sql('data', conn)

our database now contains the table named `data` with the primary key `index`. We need to
provide this information to SKompiler to have it generate the correct query:

    sql = expr.to('sqlalchemy/sqlite', key_column='index', from_obj='data')

We may now query the data:

    results = pd.read_sql(sql, conn)
    
and verify that the results match:

    assert (results.values.ravel() == m.predict(X).ravel()).all()

Note that the generated SQL expression uses names `x1`, `x2`, `x3` and `x4` to refer to the input variables.
We may have chosen different input variable names by writing:

    expr = skompile(m.predict, ['a', 'b', 'c', 'd'])

### Single-shot computation

Note that the generated SQL code splits the computation into steps using `with` expressions. In some cases
you might want to have the whole computation "inlined" into a single expression. You may achieve this by specifying
`multistage=False`:

    sql = expr.to('sqlalchemy/sqlite', multistage=False)

Note that in this case the resulting expression would typically be several times longer than the multistage version:

    len(expr.to('sqlalchemy/sqlite'))
    > 2262
    len(expr.to('sqlalchemy/sqlite', multistage=False))
    > 12973

Why so? Because, for a typical classifier (including the one used in this example)

    predict(x) = argmax(predict_proba(x))

There is, however, no single `argmax` function in SQL, hence it has to be faked using approximately the following logic:

    predict(x) = if predict_proba(x)[0] == max(predict_proba(x)) then 0
                    else if predict_proba(x)[1] == max(predict_proba(x)) then 1
                    else 2

Now, if we may not use a separate step to compute and store the output of `predict_proba`, we need to repeat the same computation verbatim during inlining.
To summarize, you should probably avoid the use of `multistage=False` in most cases.

### Other formats

By changing the first parameter of the `.to()` call you may produce output in a variety of other formats besides SQLite:

  * `sqlalchemy`: raw SQLAlchemy expression (which is a dialect-independent way of representing SQL). Jokes aside, SQL is sometimes a totally valid choice for deploying models into production.
  
     Note that generated SQL may (depending on the chosen method) include functions `exp` and `log`. If you work with SQLite, bear in mind that these functions are not supported out of the box and need to be [added separately](https://stackoverflow.com/a/2108921/318964) via `create_function`. You can find an example of how this can be done in `tests/evaluators.py` in the package source code.
  * `sqlalchemy/<dialect>`: SQL string in any of the SQLAlchemy-supported dialects (`firebird`, `mssql`, `mysql`, `oracle`, `postgresql`, `sqlite`, `sybase`). This is a convenience feature for those who are lazy to figure out how to compile raw SQLAlchemy to actual SQL.
  * `excel`: Excel formula. Ever tried dragging a random forest equation down along the table? Fun! Check out [this short screencast](https://www.youtube.com/watch?v=7vUfa7W0NpY) to see how it can be done.
  
    _NB: The screencast was recorded using a previous version, where `multistage=False` was the default_.
  * `sympy`: A SymPy expression. Ever wanted to take a derivative of your model symbolically?
  * `sympy/<lang>`: Code in the language `<lang>`, generated via SymPy. Supported values for `<lang>` are `c`, `cxx`, `rust`, `fortran`, `js`, `r`, `julia`, `mathematica`, `octave`. Note that the quality of the generated code varies depending on the model, language and the value of the `assign_to` parameter. Again, this is just a convenience feature, you will get more control by dealing with `sympy` oode printers [manually](https://www.sympy.org/scipy-2017-codegen-tutorial/). 
  
    _NB: Sympy translation does not support multistage mode at the moment, hence the resulting code will have repeated subexpressions (which can be extracted by means of Sympy itself, however)._

  * `python`: Python syntax tree (the same you'd get via `ast.parse`). This (and the following three options) are mostly useful for debugging and testing.
  * `python/code`: Python source code. The generated code will contain references to custom functions, such as `__argmax__`, `__sigmoid__`, etc. To execute the code you will need to provide these in the `locals` dictionary. See `skompiler.fromskast.python._eval_vars`.
  * `python/lambda`: Python callable function (primarily useful for debugging and testing). Equivalent to calling `expr.lambdify()`.
  * `string`: string, equivalent to `str(expr)`.

### How it works

The `skompile` procedure translates a given method into an intermediate syntactic representation (called SKompiler AST or SKAST). This representation uses a limited number of operations so it is reasonably simple to translate it into other forms.

It is important to understand the following:

 * So far this has been a fun mostly two-weekend project, hence the "compilation" of models into SKAST was only implemented for linear models, decision trees, random forest and gradient boosting.
 * In principle, SKAST's utility is not limited to `sklearn` models. Anything you translate into SKAST becomes automatically compileable to whatever output backends are implemented in `SKompiler`. Generating SKAST is rather straightforward:

       from skompiler import ast
       expr = ast.BinOp(ast.Add(), ast.Identifier('x'), ast.NumberConstant(1))
       expr.to('sqlalchemy/sqlite', 'result')
       > x + 1 as result

   Simpler expressions can be generated from strings:

       from skompiler.toskast.string import translate as fromstring
       fromstring('10 * (x + 1)')

   Conversely, you can use `repr(expr)` on any SKAST expression to dump its unformatted internal representation for examination.

 * For larger models (say, a random forest or a gradient boosted model with 500+ trees) the resulting SKAST expression tree may become deeper than Python's default recursion limit of 1000. As a result you will get a `RecursionError` when trying to traslate the model. To alleviate this, raise the system recursion limit to sufficiently high value:

       import sys
       sys.setrecursionlimit(10000)

Development
-----------

If you plan to develop or debug the package, consider installing it by running:

    $ pip install -e .[dev]

from within the source distribution. This will install the package in "development mode" and include extra dependencies, useful for development.

You can then run the tests by typing

    $ py.test
    
at the root of the source distribution.

Contributing
------------

Feel free to contribute or report issues via Github:

 * https://github.com/konstantint/SKompiler


Copyright & License
-------------------

Copyright: 2018, Konstantin Tretyakov.
License: MIT
