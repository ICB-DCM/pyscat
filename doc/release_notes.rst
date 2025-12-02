Release notes
=============

pyscat-0.0.1 (2025-12-02)
-------------------------

PyScat started from the scatter search implementation included in
pyPESTO 0.5.7.

Most relevant changes since pyPESTO 0.5.7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The deprecated ``startpoint_method`` has been removed from
  ``SacessOptimizer.minimize`` and ``ESSOptimizer.minimize``.
  ``Problem.startpoint_method`` is now used instead.
* Fixed a bug in counting of function evaluations that lead to double counting
  of some function evaluations in eSS and saCeSS optimizers, potentially
  resulting in premature termination if a function evaluation budget was set.
* Fixed a bug in the ranking of candidates for local optimization startpoints
  in eSS and saCeSS optimizers.
* Updated/extended documentation and examples.
* Initialization of ``SacessOptimizer`` is now problem-dependent, i.e., a
  ``Problem`` instance has to be provided at initialization instead of at
  ``minimize`` time.
  This simplifies changing local optimizers, for which
  ``SacessOptimizer.set_local_optimizer`` was added.
* If the provided objective function does not provide gradient information,
  no local optimizer is used by default.
  This may change in future (minor) releases.
* Added experimental functionality to record all function evaluations during
  optimization, or the k-best ones, or those below a certain threshold
  based on the best objective function value found so far.
  This is intended to construct parameter ensembles for uncertainty
  quantification. See example in the documentation.


Migrating from pyPESTO 0.5.7 to PyScat 0.0.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example of migrating from pyPESTO 0.5.7 to PyScat 0.0.1 is shown below:

.. code-block:: python

    # Old pyPESTO 0.5.7 code
    from pypesto.optimize import SacessOptimizer
    from pypesto.optimize.ess import get_default_ess_options

    problem = Problem(...)  # define your problem here

    ess_init_args = get_default_ess_options(
        num_workers=12,
        dim=problem.dim,
        local_optimizer=False,
    )
    optimizer = SacessOptimizer(
        ess_init_args=ess_init_args,
        max_walltime_s=5,
        sacess_loglevel=logging.WARNING
    )
    result = optimizer.minimize(problem)


.. code-block:: python

    # New PyScat 0.0.1 code
    from pyscat import SacessOptimizer

    problem = Problem(...)  # define your problem here

    optimizer = SacessOptimizer(
        problem=problem,
        num_workers=12,
        max_walltime_s=5,
        sacess_loglevel=logging.WARNING
    )
    optimizer.set_local_optimizer(None)
    result = optimizer.minimize()
