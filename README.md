[![PyPI - Version](https://img.shields.io/pypi/v/pyscat)](https://pypi.org/project/pyscat/)

# PyScat

*A Python library for scatter-search metaheuristics.*

> **Note**
> This package is still under development. Documentation is incomplete, and the API may change without notice. The PyPI package is currently only a stub and does not contain the full functionality.
> For now, please use the scatter search implementation in
> [pyPESTO](https://pypesto.readthedocs.io/en/latest/).

PyScat currently implements two scatter-search algorithms:

* **Enhanced Scatter Search (eSS)**
  based on the work by Egea et al. (2007)
  ([DOI:10.1021/ie801717t](https://doi.org/10.1021/ie801717t))
* **Self-Adaptive Cooperative enhanced Scatter Search (saCeSS)**
  based on the work by Penas et al. (2017)
  ([DOI:10.1186/s12859-016-1452-4](https://doi.org/10.1186/s12859-016-1452-4))

PyScat currently builds on top of the
[PyPESTO](https://pypesto.readthedocs.io/en/latest/) framework for parameter
estimation and leverages its problem definition and optimizer interfaces.

## 📖 Documentation

The documentation will hopefully soon be available at
[pyscat.readthedocs.io](https://pyscat.readthedocs.io/).

## 📦 Installation

Once available on PyPI:

```bash
pip install pyscat
```
