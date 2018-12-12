SKompiler: Translate trained SKLearn models to executable code in other languages
================================================================================

The package provides a tool for transforming trained SKLearn models into other forms, such as SQL queries, Excel formulas or Sympy expressions (which, in turn, can be translated to code in a variety of languages, such as C, Javascript, Rust, Julia, etc).

Installation
------------

The simplest way to install the package is via `pip`:

    $ pip install SKompiler[full]


Note that the `[full]` option includes the installations of `sympy`, `sqlalchemy` and `astor`, which are necessary if you plan to convert `SKompiler`'s expressions to `sympy` expressions (which, in turn, can be compiled to many other languages) or to SQLAlchemy expressions (which can be further translated to different SQL dialects) or to Python source code. If you do not need this functionality (say, you only need the raw `SKompiler` expressions or the conversions implemented in `SKompiler` code without relying on `sympy`, such as SQL or Excel), you may avoid the forced installation of optional dependencies by simply writing

    $ pip install SKompiler

(you are free to install some of the required extra dependencies, of course)

Usage
-----

### Introductory example

Let us start by walking through a simple example. We begin by training a model on a simple dataset, e.g.:

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_iris(True)
    m = RandomForestClassifier(n_estimators=3, max_depth=3).fit(X, y)

Suppowe we need to express the logic of `m.predict` in SQLite. Here is how we can achieve that:

    from skompiler import skompile
    expr = skompile(m.predict)
    sql = expr.to('sqlalchemy/sqlite')

Voila, the value of the `sql` variable is a super-long expression which looks like

    CASE WHEN ((CASE WHEN (x3 <= 2.449999988079071) THEN 1.0 ELSE CASE WHEN
    ... 100 lines or so ...
    THEN 1 ELSE 2 END as y

It corresponds to the `m.predict` computation. Let us check how we can use it in a query.
First import the data into an in-memory SQLite database:

    import sqlalchemy as sa
    import pandas as pd
    conn = sa.create_engine('sqlite://').connect()
    pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4']).to_sql('data', conn)

Now query the generated expression:

    results = pd.read_sql('select {0} from data'.format(sql), conn)

and verify that the results match:

    assert (results.values.ravel() == m.predict(X).ravel()).all()

Note that the generated SQL expression uses names `x1`, `x2`, `x3` and `x4` to refer to the input variables. You may choose different input variable names by writing:

    skompile(m.predict, ['a', 'b', 'c', 'd']).to('sqlalchemy/sqlite')

### Multiple outputs

Now let us try to generate code for `m.predict_proba`:

    expr = skompile(m.predict_proba)
    expr.to('sqlalchemy/sqlite')

The generated query is different from the previous one. Firstly, it is of the form

    ... as y1, ... as y2, ... as y3

The reason for that is that `m.predict_proba` produces three values - the probabilities of each class, and this is reflected in the SQL. You may, of course, provide different names to the outputs instead of `y1`,`y2`,`y3`:

    expr.to('sqlalchemy/sqlite', assign_to=['a','b','c'])

You may obtain a list of three separate expressions without the `as ..` parts at all:

    expr.to('sqlalchemy/sqlite', assign_to=None)

or request only the probability of the first class as a single `... as y2` expression:

    expr.to('sqlalchemy/sqlite', component=1, assign_to='y2')

### Other formats

By changing the first parameter of the `.to()` call you may produce output in a variety of other formats besides SQLite:

  * `sqlalchemy`: raw SQLAlchemy expression (which is a dialect-independent way of representing SQL). Jokes aside, SQL is sometimes a totally valid choice for deploying models into production.
  * `sqlalchemy/<dialect>`: SQL in any of the SQLAlchemy-supported dialects (`firebird`, `mssql`, `mysql`, `oracle`, `postgresql`, `sqlite`, `sybase`). This is a convenience feature for those who are lazy to figure out how to compile raw SQLAlchemy to actual SQL.
  * `excel`: Excel formula. Ever tried dragging a random forest equation down along the table? Fun! Due to its 8196-character limit on the formula length, however, Excel will not handle forests larger than `n_estimators=30` with `max_depth=5` or so, unfortunately.
  * `sympy`: A SymPy expression. Ever wanted to take a derivative of your model symbolically?
  * `sympy/<lang>`: Code in the language `<lang>`, generated via SymPy. Supported values for `<lang>` are `c`, `cxx`, `rust`, `fortran`, `js`, `r`, `julia`, `mathematica`, `octave`. Note that the quality of the generated code varies depending on the model, language and the value of the `assign_to` parameter. 
  * `python`: Python syntax tree (the same you'd get via `ast.parse`). This (and the following three options) are mostly useful for debugging and testing.
  * `python/code`: Python source code.
  * `python/lambda`: Python callable function (primarily useful for debugging and testing). Equivalent to calling `expr.lambdify()`.
  * `string`: string, equivalent to `str(expr)`.

### How it works

The `skompile` procedure translates a given method into an intermediate syntactic representation (called SKompiler AST or SKAST). This representation uses a limited number of operations so it is reasonably simple to translate it into other forms.

It is important to understand the following:

 * So far this has been a fun mostly single-weekend project, hence the "compilation" of models into SKAST was only implemented for linear models, decision trees, random forest and gradient boosting.
 * In principle, SKAST's utility is not limited to `sklearn` models. Anything you translate into SKAST becomes automatically compileable to whatever output backends are implemented in `SKompiler`. Generating SKAST is rather straightforward:

       from skompiler import ast
       expr = ast.BinOp(ast.Add(), ast.Identifier('x'), ast.NumberConstant(1))
       expr.to('sqlalchemy/sqlite', 'result')
       > x + 1 as result

 * At the moment `skompiler` transforms models into *expressions*, and this may affect the complexity of the output. You might have noted that in the introductory example above the SQL code for `predict` was significantly longer than the code for `predict_proba`. Why so? Because, essentially

       predict(x) = argmax(predict_proba(x))

   There is, however, no single `argmax` function in SQL, hence it has to be faked using approximately the following logic:

       predict(x) = if predict_proba(x)[0] == max(predict_proba(x)) then 0
                    else if predict_proba(x)[1] == max(predict_proba(x)) then 1
                    else 2

   Note that the values of `predict_proba` in this expression must be expanded (and thus the computation repeated) multiple times.

   This problem could be overcome by performing computation in steps - first saving the values of `predict_proba`, then finding the argmax - in SQL this could be implemented via CTE-s. However, this is not how `SKompiler` works, at the moment, so enjoy the super-long SQLs. For example, the SQLite expression corresponding to `RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y).predict` is about 756KB-long (surprisingly, SQLite manages to parse and execute it successfully and rather quickly. Don't bother compiling it to SymPy or feeding into Excel, though).


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
