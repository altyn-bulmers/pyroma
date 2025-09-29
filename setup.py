from setuptools import setup, find_packages

setup(
  name = 'roma-analysis',            
  version = '0.2.3',      
  license='GPL-3.0',        
  description = 'Representation and Quantification of Module Activity for bulk and single cell transcriptomics in python',   
  author = 'Altynbek Zhubanchaliyev',                  
  author_email = 'altynbek.zhubanchaliyev@curie.fr',      
  url = 'https://github.com/altyn-bulmers',  
  download_url = 'https://github.com/altyn-bulmers/pyroma/archive/refs/tags/0.2.3.tar.gz',    
  keywords = ['python', 'bioinformatics', 'machine-learning', 
              'pathway-activity', 'transcriptomics', 'rna-seq-analysis', 'single-cell-rna-seq', 
              'pathway-analysis', 'pathway-enrichment-analysis' 
              ],
  packages=find_packages(),
  package_data={
        'pyroma.genesets': ['*.gmt'],
        'pyroma.datasets': ['*.h5ad', '*.tsv', '*.csv'],
    },
  include_package_data=True,   
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
  python_requires='>=3.7',
)