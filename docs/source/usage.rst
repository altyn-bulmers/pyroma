Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install roma-analysis

Or from source:

.. code-block:: bash

   git clone https://github.com/altyn-bulmers/pyroma.git
   cd pyroma
   pip install -e .

Example
-------
.. code-block:: python

   import pyroma
   roma = pyroma.ROMA()
   roma.adata = adata
   roma.gmt   = "h.all.v2023.2.Hs.symbols.gmt"
   roma.compute()
   roma.adata.uns['ROMA_active_modules']