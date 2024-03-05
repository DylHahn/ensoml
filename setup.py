from setuptools import setup


dependencies = [
    'h5netcdf', 'h5py', 'matplotlib',
    'neuralforecast', 'numpy', 'pandas',
    'plotly', 'plotly-express',
    'scikit-learn', 'scipy',  'xarray',
    ]

setup(
    name='enso-nf',
    version='1.0',
    description='A package to analyze the predictive strength of varying ML models wrt ENSO data',
    author='Jesus Perez Cuarenta',
    author_email='jesus.perezcuarenta@gmail.com',
    python_requires='~=3.8',
    install_requires=dependencies,
    )