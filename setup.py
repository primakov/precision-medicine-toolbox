from distutils.core import setup
setup(
  name = 'precision-medicine-toolbox',         # How you named your package folder (MyLib)
  packages = ['precision-medicine-toolbox'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='bsd-3-clause',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Precision medicine tool-box for medical imaging research',   # Give a short description about your library
  author = 'sergey primakov & lisa lavrova',                   # Type in your name
  author_email = 'primakov@bk.ru',      # Type in your E-Mail
  url = 'https://github.com/primakov/precision-medicine-toolbox',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['medical imaging research', 'dicom', 'radiomics', 'statistical analysis', 'features'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)