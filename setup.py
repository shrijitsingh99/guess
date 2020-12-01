from setuptools import find_packages, setup

setup(
    name='guess',
    version='0.0.0',
    description='GUESs Library',
    package_dir={'': 'python'},
    packages=find_packages('python', exclude=['test', 'test.*']),
    install_requires=[
        'dataclasses>=0.6',
        'pytest>=4.0.1',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=2.2']
    }
)
