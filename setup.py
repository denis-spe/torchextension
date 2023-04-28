# === Glory Be To God ====

from setuptools import setup

setup(
   name='torchextension',
   version='0.0.1',
   author='denis-spe',
   author_email='denisbrian07@gmail.com',
   packages=['torchextension', 'tests'],
   url='http://pypi.python.org/pypi/torchextension/',
   license='LICENSE.txt',
   description='torch sequential extension',
   long_description=open('README.txt').read(),
   install_requires=[
      'torchinfo',
      'tqdm'
   ],
)