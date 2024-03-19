from setuptools import setup
required = None
with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(  
    name='LUCI',
    version='1.1',
    author='Carter Rhea',
    description='general purpose fitting pipeline built specifically with SITELLE IFU data cubes in mind',
    long_description='Much more information can be found at the documentation: https://crhea93.github.io/LUCI/index.html',
    url='https://github.com/crhea93/LUCI',
    keywords='development, setup, setuptools',
    python_requires='>=3.7, <4',
    packages=['distutils', 'distutils.command'],
    install_requires=required
)