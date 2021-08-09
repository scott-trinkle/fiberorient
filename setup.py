from setuptools import setup
from os import path

# read the contents of README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DEPENDENCIES = ['numpy', 'scikit-image',
                'scipy', 'scikit-learn', 'dipy']

setup(name='fiberorient',
      version='0.1',
      description='Tools for 3D structure tensor analysis',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/scott-trinkle/fiberorient',
      author='Scott Trinkle',
      author_email='tscott.trinkle@gmail.com',
      license='MIT',
      packages=['fiberorient'],
      package_dir={'fiberorient': 'fiberorient'},
      package_data={'fiberorient': ['data/*']},
      install_requires=DEPENDENCIES)
