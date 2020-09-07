.. _DetailedInstallationGuide:

Detailed Installation guide
---------------------------

To install myFM you have to prepare the following:

* a decent C++ compiler with C++11 support.
    myFM is a C++ extension for Python: it is a shared library built using the C++ compiler
    with the help of the elegant interface library `Pybind11 <https://github.com/pybind/pybind11>`_ (version >=2.5.0).
    Therefore, you need 

* (Optional but highly recommended) Internet Environment.
    myFM makes extensive use of `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_'s sparse matrix,
    and it downloads the right version of Eigen during the installation
    if you don't specify one.
    If you want to use Eigen which you have prepared yourself, you can use it like: ::

        EIGEN3_INCLUDE_DIR=/path/to/eigen pip instal myfm


Other python dependencies (specified in ``setup.py`` ) include

* `numpy <https://numpy.org/>`_ >= 1.11 and `scipy <https://www.scipy.org/scipylib/index.html>`_ >=1.0 for manipulating dense or sparse matrices.
* `tqdm <https://github.com/tqdm/tqdm>`_ >= 4.0 for an elegant progress bar.

