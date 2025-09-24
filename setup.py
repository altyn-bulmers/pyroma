
from distutils.core import setup
setup(
  name = 'roma-analysis',         
  packages = ['pyroma'],   
  version = '0.2.0',      
  license='GPL-3.0',        
  description = 'Representation and Quantification of Module Activity for bulk and single cell transcriptomics in python',   
  author = 'Altynbek Zhubanchaliyev',                  
  author_email = 'altynbek.zhubanchaliyev@curie.fr',      
  url = 'https://github.com/altyn-bulmers',  
  download_url = 'https://github.com/altyn-bulmers/pyroma/archive/refs/tags/0.2.0.tar.gz',    
  keywords = ['python', 'bioinformatics', 'machine-learning', 
              'pathway-activity', 'transcriptomics', 'rnaseq', 'single-cell-rna-seq', 
              ],   
  install_requires=[            
          'scanpy',
          'scikit-learn',
          'numpy',
          'matplotlib',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   
    'Programming Language :: Python :: 3',      
  ],
)