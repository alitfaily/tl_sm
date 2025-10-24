from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_file(filename):
    here = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(here, filename)
    if os.path.exists(filepath):
        with open(filepath, encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt if it exists
def read_requirements(filename='requirements.txt'):
    here = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(here, filename)
    if os.path.exists(filepath):
        with open(filepath, encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='tlsm',
    version='0.1.0',
    author='Ali Tfaily',
    description='Transfer Learning in Surrogate Modeling',
    url='https://github.com/alitfaily/tl_sm',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    
    # Core dependencies - customize based on your actual requirements
    install_requires=read_requirements() if os.path.exists('requirements.txt') else [
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'smt>=2.0.1',
        'matplotlib>=3.4.0',
    ],
    


    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/alitfaily/tl_sm/issues',
        'Source': 'https://github.com/alitfaily/tl_sm',
    },
    
    keywords='transfer-learning surrogate-modeling machine-learning deep-learning',
    zip_safe=False,
)
