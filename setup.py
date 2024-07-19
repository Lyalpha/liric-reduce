from setuptools import setup, find_packages

setup(
    name='liric-reduce',
    version='1.0.0',
    packages=find_packages(),
    license='MIT',
    author='Joe Lyman',
    author_email='joedlyman@gmail.com',
    description='Reduction package for Liverpool Telescope LIRIC data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'astropy',
        'image_registration',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Your License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    entry_points={
        'console_scripts': [
            'liric-reduce=liric_reduce.reduce:run',
        ],
    },
)