.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. We are especially interested in suggestions
on how to speed up the algorithm.

You can contribute in several ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/csprock/bmdcluster/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/csprock/bmdcluster/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)


Algorithmic Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~

If you are interested in improving the performance of the algorithm, follow the instructions under Submit Feedback and tag the issue as an enhancement. The maintainers
will contact you. Before submitting a PR, make sure the improvements are consistent with the current user interface APIs and unit tests. Also make sure all your enhancements
pass the current unit tests. 

Getting Started
---------------

Ready to contribute? Here's how to set up `bmdcluster` for local development.

1. Fork the `bmdcluster` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/bmdcluster.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv bmdcluster
    $ cd bmdcluster/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests

    $ flake8 bmdcluster tests
    $ python setup.py test or py.test


6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests or pass existing ones.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should work for Python 3.4, 3.5, 3.6, 3.7 and for PyPy. Check
   https://travis-ci.org/csprock/bmdcluster/pull_requests
   and make sure that the tests pass for all supported Python versions.





