'''
SKompiler: Library for converting trained SKLearn models into abstract expressions suitable
for further compilation into executable code in various languages.

Author: Konstantin Tretyakov
License: MIT
'''

from setuptools import setup, find_packages

extra_requires = []
try:
    from functools import singledispatch #pylint: disable=unused-import
except ImportError:
    # Python 3.3-
    extra_requires.append("singledispatch")

setup(name='SKompiler',
      version=[ln for ln in open("skompiler/__init__.py") if ln.startswith("__version__")][0].split("'")[1],
      description="Library for compiling trained SKLearn models into abstract expressions "
                  "suitable for further compilation into executable code in various languages.",
      long_description=open("README.md", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development :: Code Generators',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
      ],
      keywords='sklearn datascience modelling deployment',
      author='Konstantin Tretyakov',
      author_email='kt@ut.ee',
      url='https://github.com/konstantint/SKompiler',
      license='MIT',
      packages=find_packages(exclude=["examples", "tests"]),
      include_package_data=True,
      zip_safe=True,
      install_requires=["scikit-learn"] + extra_requires,
      extras_require={
          "full": ["sympy", "sqlalchemy", "astor >= 0.6"],
          "test": ["sympy", "sqlalchemy", "astor >= 0.6", "pytest", "pandas", "keras", "theano"],
          "dev": ["sympy", "sqlalchemy", "astor >= 0.6", "pytest", "pandas", "keras", "theano",
                  "pylint", "jupyter", "twine"],
      }
     )
