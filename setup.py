from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name='GOSTRocks',
    packages=['GOSTRocks'],
    install_requires=[
        'rasterio',
        'geopandas',
        'pandas',
        'numexpr > 2.6.8',
        'numpy',
        'pyproj',
        'ogr',
        'seaborn',
        'boto3',
        'botocore',
        'contextily',
        'matplotlib',
        'tqdm',
        'xarray',
        'osmnx',
        'affine',
        'PyOpenSSL >= 23.2'  

    ],
    version='0.1.0',
    description='Miscellaneous geospatial functions concerning vector, raster, and network analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bpstewar/gostrocks",    
    author='Benjamin P. Stewart',    
    package_dir= {'':'src'}
)