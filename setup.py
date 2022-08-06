from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
  name = 'precision-medicine-toolbox',         # How you named your package folder (MyLib)
  packages = ['pmtool'],   # Chose the same as "name"
  version = '0.9',      # Start with a small number and increase it with every change you make
  license='bsd-3-clause',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Precision medicine tool-box for medical imaging research',   # Give a short description about your library
  long_description=long_description,
  long_description_content_type = 'text/markdown',
  author = 'sergey primakov & lisa lavrova',                   # Type in your name
  author_email = 'primakov@bk.ru',      # Type in your E-Mail
  url = 'https://github.com/primakov/precision-medicine-toolbox',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/primakov/precision-medicine-toolbox/archive/refs/tags/0.9.tar.gz',    # I explain this later on
  keywords = ['medical imaging research', 'dicom', 'radiomics', 'statistical analysis', 'features'],   # Keywords that define your package best
  install_requires=[
    'SimpleITK',
    'PyWavelets',
    'pykwalify',
    'six',
    'tqdm',
    'pydicom',
    'pandas',
    'pyradiomics',
    'scikit-image',
    'ipywidgets',
    'matplotlib',
    'Pillow',
    'scikit-learn',
    'scipy',
    'plotly',
    'mkdocstrings',
    'statsmodels',
    'opencv-python'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
