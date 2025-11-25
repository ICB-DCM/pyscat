Development
===========


Development setup
-----------------

We use `pre-commit <https://pre-commit.com/>`_ to run linters and formatters on
the codebase. To enable pre-commit hooks in your development environment, run:

.. code-block:: bash

    pip install pre-commit
    pre-commit install


Running tests
-------------

We use `pytest <https://docs.pytest.org/en/stable/>`_ for testing.

To run the test suite, execute the following command in the project root
directory:

.. code-block:: bash

    pytest

Building documentation
----------------------

The documentation is built using `Sphinx <https://www.sphinx-doc.org/>`__.

To build the documentation locally, navigate to the ``doc/`` directory and run:

.. code-block:: bash

    make html

The generated HTML files will be located in the ``doc/_build/html/`` directory.
