About PyScat
============

.. image:: https://zenodo.org/badge/1103842718.svg
  :target: https://doi.org/10.5281/zenodo.17787407
  :alt: Zenodo DOI

PyScat is a Python library implementing scatter-search metaheuristics
for global optimization.

.. note::

   This package is still under development.
   Documentation is incomplete, and the API may change without notice.
   For now, please use the scatter search implementations available in
   `pyPESTO <https://pypesto.readthedocs.io/en/latest/>`_.

Installation
------------

Releases from `PyPI <https://pypi.org/project/pyscat/>`_:

.. code-block:: bash

    pip install pyscat

The latest development version from GitHub:

.. code-block:: bash

    pip install git+https://github.com/ICB-DCM/pyscat.git#egg=pyscat

Algorithms
----------

* Enhanced Scatter Search (eSS)

  Based on `Egea et al. (2007) <https://doi.org/10.1021/ie801717t>`_

* Self-Adaptive Cooperative enhanced Scatter Search (saCeSS)

  Based on `Penas et al. (2017) <https://doi.org/10.1186/s12859-016-1452-4>`_
