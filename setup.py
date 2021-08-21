from setuptools import setup
from os import path

# read the contents of README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DEPENDENCIES = ['numpy', 'scipy', 'scikit-learn', 'dipy', 'fury']

setup(name='fiberorient',
      version='0.2',
      description='Tools for 3D structure tensor analysis',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/scott-trinkle/fiberorient',
      download_url='https://github.com/scott-trinkle/fiberorient/archive/refs/tags/v0.2.tar.gz',
      author='Scott Trinkle',
      author_email='tscott.trinkle@gmail.com',
      license='MIT',
      keywords=['structure tensor', 'orientation', 'ODF'],
      packages=['fiberorient'],
      package_dir={'fiberorient': 'fiberorient'},
      package_data={'fiberorient': ['data/*']},
      install_requires=DEPENDENCIES,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
      ])
