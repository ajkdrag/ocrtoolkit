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

Supports:

- DocTR
- PaddleOCR
- Ultralytics
- Google Cloud Vision


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
    

The `API Reference <https://ajkdrag.github.io/ocrtoolkit/>`_ provides API-level documentation.



