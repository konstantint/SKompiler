SKompiler: Translate trained SKLearn models to executable code in other languages
================================================================================

[![Build Status](https://travis-ci.org/konstantint/SKompiler.svg?branch=master)](https://travis-ci.org/konstantint/SKompiler)

The package provides a tool for transforming trained SKLearn models into other forms, such as SQL queries, Excel formulas, Portable Format for Analytics (PFA) files or Sympy expressions (which, in turn, can be translated to code in a variety of languages, such as C, Javascript, Rust, Julia, etc).

Requirements
------------

 - Python 3.5 or later

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

Let us start by walking through an introductory example. We begin by training a model on a small dataset:

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_iris(True)
    m = RandomForestClassifier(n_estimators=3, max_depth=3).fit(X, y)

Suppose we need to express the logic of `m.predict` in SQLite. Here is how we can achieve that:

    from skompiler import skompile
    expr = skompile(m.predict)
    sql = expr.to('sqlalchemy/sqlite')

Voila, the value of the `sql` variable is a query, which would compute the value of `m.predict` in pure SQL:

    WITH _tmp1 AS
    (SELECT .... FROM data)
    _tmp2 AS
    ( ... )
    SELECT ... from _tmp2 ...

Let us import the data into an in-memory SQLite database to test the generated query:

    import sqlalchemy as sa
    import pandas as pd
    conn = sa.create_engine('sqlite://').connect()
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4']).reset_index()
    df.to_sql('data', conn)

Our database now contains the table named `data` with the primary key `index`. We need to provide this information to SKompiler to have it generate the correct query:

    sql = expr.to('sqlalchemy/sqlite', key_column='index', from_obj='data')

We can now query the data:

    results = pd.read_sql(sql, conn)
    
and verify that the results match:

    assert (results.values.ravel() == m.predict(X).ravel()).all()

Note that the generated SQL expression uses names `x1`, `x2`, `x3` and `x4` to refer to the input variables.
We could have chosen different input variable names by writing:

    expr = skompile(m.predict, ['a', 'b', 'c', 'd'])

### Single-shot computation

Note that the generated SQL code splits the computation into sequential steps using `with` expressions. In some cases you might want to have the whole computation "inlined" into a single expression. You can achieve this by specifying
`multistage=False`:

    sql = expr.to('sqlalchemy/sqlite', multistage=False)

Note that in this case the resulting expression would typically be several times longer than the multistage version:

    len(expr.to('sqlalchemy/sqlite'))
    > 2262
    len(expr.to('sqlalchemy/sqlite', multistage=False))
    > 12973

Why so? Because, for a typical classifier (including the one used in this example)

    predict(x) = argmax(predict_proba(x))

There is, however, no single `argmax` function in SQL, hence it has to be faked using the following logic:

    predict(x) = if predict_proba(x)[0] == max(predict_proba(x)) then 0
                    else if predict_proba(x)[1] == max(predict_proba(x)) then 1
                    else 2

If SKompiler is not alowed to use a separate step to store the intermediate `predict_proba` outputs, it is forced to inline the same computation verbatim multiple times. To summarize, you should probably avoid the use of `multistage=False` in most cases.

### Other formats

By changing the first parameter of the `.to()` call you may produce output in a variety of other formats besides SQLite:

  * `sqlalchemy`: raw SQLAlchemy expression (which is a dialect-independent way of representing SQL). Jokes aside, SQL is sometimes a totally valid choice for deploying models into production.
  
     Note that generated SQL may (depending on the chosen model and method) include functions `exp`, `log` and `sqrt`, which are not supported out of the box in SQLite. If you work with SQLite, you will need to [add them separately](https://stackoverflow.com/a/2108921/318964) via `create_function`. You can find an example of how this can be done in `tests/evaluators.py` in the SKompiler's source code.
  * `sqlalchemy/<dialect>`: SQL string in any of the SQLAlchemy-supported dialects (`firebird`, `mssql`, `mysql`, `oracle`, `postgresql`, `sqlite`, `sybase`). This is a convenience feature for those who are lazy to figure out how to compile raw SQLAlchemy to actual SQL.
  * `excel`: Excel formula. Ever tried dragging a random forest equation down along the table? Fun! Check out [this short screencast](https://www.youtube.com/watch?v=7vUfa7W0NpY) to see how it can be done.
  
    _NB: The screencast was recorded using a previous version, where `multistage=False` was the default option_.
  * `pfa`: A dict with [PFA](http://dmg.org/pfa/) code.
  * `pfa/json` or `pfa/yaml`: PFA code as a JSON or YAML string for those who are lazy to write `json.dumps` or `yaml.dump`. PyYAML should be installed in the latter case, of course.
  * `sympy`: A SymPy expression. Ever wanted to take a derivative of your model symbolically?
  * `sympy/<lang>`: Code in the language `<lang>`, generated via SymPy. Supported values for `<lang>` are `c`, `cxx`, `rust`, `fortran`, `js`, `r`, `julia`, `mathematica`, `octave`. Note that the quality of the generated code varies depending on the model, language and the value of the `assign_to` parameter. Again, this is just a convenience feature, you will get more control by dealing with `sympy` code printers [manually](https://www.sympy.org/scipy-2017-codegen-tutorial/). 
  
    _NB: Sympy translation does not support multistage mode at the moment, hence the resulting code will have repeated subexpressions (which can be extracted by means of Sympy itself, however)._

  * `python`: Python syntax tree (the same you'd get via `ast.parse`). This (and the following three options) are mostly useful for debugging and testing.
  * `python/code`: Python source code. The generated code will contain references to custom functions, such as `__argmax__`, `__sigmoid__`, etc. To execute the code you will need to provide these in the `locals` dictionary. See `skompiler.fromskast.python._eval_vars`.
  * `python/lambda`: Python callable function (primarily useful for debugging and testing). Equivalent to calling `expr.lambdify()`.
  * `string`: string, equivalent to `str(expr)`.

### Other models

So far this has been a fun two-weekends-long project, hence translation is implemented for a limited number of models. The most basic ones (linear models, decision trees, forests, gradient boosting, PCA, KMeans, MLP, Pipeline and a couple of preprocessors) are covered, however, and this is already sufficient to compile nontrivial constructions. For example:

    m = Pipeline([('scale', StandardScaler()),
                  ('dim_reduce', PCA(6)),
                  ('cluster', KMeans(10)),
                  ('classify', MLPClassifier([5, 4], 'tanh'))])

Even though this particular example probably does not make much sense from a machine learning perspective, it would happily compile both to Excel and SQL forms none the less.

Rudimentary support for theano-backed Keras models (covering the basic `Dense` layer for now) is also available. The following would work, for example:

    import os
    os.environ['KERAS_BACKEND'] = 'theano'
    
    from keras.models import Sequential
    from keras.layers import Dropout, Dense
    m = Sequential([Dense(3, activation='tanh', input_shape=(4,)),
                    Dense(3, activation='linear'),
                    Dropout(0.5),
                    Dense(3, activation='sigmoid')])

    from skompiler import skompile
    skompile(m.predict)

### How it works

The `skompile` procedure translates a given method into an intermediate syntactic representation (called SKompiler AST or SKAST). This representation uses a limited number of operations so it is reasonably simple to translate it into other forms.

In principle, SKAST's utility is not limited to `sklearn` models. Anything you translate into SKAST becomes automatically compileable to whatever output backends are implemented in `SKompiler`. Generating raw SKAST is quite straightforward:

    from skompiler.dsl import ident, const
    expr = const([[1,2],[3,4]]) @ ident('x', 2) + 12
    expr.to('sqlalchemy/sqlite', 'result')
    > SELECT 1 * x1 + 2 * x2 + 12 AS result1, 3 * x1 + 4 * x2 + 12 AS result2 
    > FROM data

You can use `repr(expr)` on any SKAST expression to dump its unformatted internal representation for examination or `str(expr)` to get a somewhat-formatted view of it.

It is important to note, that for larger models (say, a random forest or a gradient boosted model with 500+ trees) the resulting SKAST expression tree may become deeper than Python's default recursion limit of 1000. As a result some translators may produce a `RecursionError` when processing such expressions. This can be solved by raising the system recursion limit to sufficiently high value:

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
