ocrtoolkit
############

Parse bank cheques


Quickstart
==========

ocrtoolkit is available on PyPI and can be installed with `pip <https://pypi.org/project/ocrtoolkit/>`_.

.. code-block:: console

    $ pip install ocrtoolkit

After installing ocrtoolkit you can use it like any other Python module.

Here is a simple example:

.. code-block:: python

    from ocrtoolkit.models import UL_YOLOV8
    from ocrtoolkit.datasets import FileDS
    from ocrtoolkit.core import detect

    ds = FileDS("some_images_dir")
    mini_ds = ds.sample()
    model = UL_YOLOV8()

    l_results = detect(model, mini_ds, stream=False):
    

The `API Reference <http://ocrtoolkit.readthedocs.io>`_ provides API-level documentation.
