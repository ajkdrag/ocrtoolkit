|docs_badge| |pypi_badge|

.. |docs_badge| image:: https://img.shields.io/github/deployments/ajkdrag/ocrtoolkit/github-pages?label=docs
   :alt: GitHub-Pages deployment status
   :target: https://ajkdrag.github.io/ocrtoolkit/ocrtoolkit/

.. |pypi_badge| image:: https://img.shields.io/pypi/v/ocrtoolkit?style=flat&color=green
   :alt: PyPI - Version
   :target: https://pypi.org/project/ocrtoolkit/


ocrtoolkit
############

Versatile Python package for seamlessly integrating and experimenting with various OCR and Object Detection frameworks.


.. contents::
   :local:

Quickstart
==========

ocrtoolkit is available on PyPI and can be installed with `pip <https://pypi.org/project/ocrtoolkit/>`_.
Supports integrations with:

- `DocTR <https://github.com/mindee/doctr/tree/main>`_
- `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_
- `Ultralytics <https://github.com/ultralytics/ultralytics>`_
- `Google Cloud Vision <https://cloud.google.com/python/docs/reference/vision/latest>`_


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

    l_results = detect(model, mini_ds, stream=False)
    

Documentation
==============

The `API Reference <https://ajkdrag.github.io/ocrtoolkit/>`_ provides API-level documentation

Notebooks
==========
Refer `notebooks <https://github.com/ajkdrag/ocrtoolkit/tree/master/notebooks>`_ for examples on how to use the modules.

.. include:: CHANGELOG.rst

