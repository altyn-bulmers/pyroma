
from distutils.core import setup
setup(
  name = 'roma-analysis',         # How you named your package folder (MyLib)
  packages = ['pyroma'],   # Chose the same as "name"
  version = '0.1.3',      # Start with a small number and increase it with every change you make
  license='GPL-3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Representation and Quantification of Module Activity for bulk and single cell transcriptomics in python',   # Give a short description about your library
  author = 'Altynbek Zhubanchaliyev',                   # Type in your name
  author_email = 'altynbek.zhubanchaliyev@curie.fr',      # Type in your E-Mail
  url = 'https://github.com/altyn-bulmers',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/altyn-bulmers/pyroma/archive/refs/tags/0.1.3.tar.gz',    # I explain this later on
  keywords = ['python', 'bioinformatics', 'machine-learning', 
              'pathway-activity', 'transcriptomics', 'rnaseq', 'single-cell-rna-seq', 
              ],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'scanpy',
          'scikit-learn',
          'numpy',
          'matplotlib',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
  ],
)