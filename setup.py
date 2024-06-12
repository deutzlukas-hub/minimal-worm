from setuptools import setup, find_packages

setup(
    name='minimal-worm',
    version='0.1.0',
    author='Lukas Deutz',
    author_email='scld@leeds.ac.uk',
    description=('Implements an active intertialess viscoelastic Cosserat rod immersed' 
    'in a Newtonian fluid to simulate slender undulating micro-swimmers'),
    packages=find_packages(),
)


