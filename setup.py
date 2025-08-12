from setuptools import setup, find_packages

setup(
    name='Genesis',  # Package name
    version='0.1.0',  # Version
    author='',  # Author
    author_email='',  # Author email
    description='Gensis is a lightweight deep learning framework written from scratch in Python, with Triton as its backend for high-performance computing.',  # Short description
    long_description=open('README.md').read(),  # Long description, usually read from README file
    long_description_content_type='text/markdown',  # Long description format
    url='https://github.com/phonism/genesis',  # Project homepage
    packages=find_packages(),  # Automatically find all packages in the project
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # License
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Python version requirement
    install_requires=[
        "numpy",
        "torch"
    ],
    entry_points={
        'console_scripts': [
            # 'your_command=your_package.module:function',
        ],
    },
    include_package_data=True,  # Include data files in package
    package_data={
        # '': ['*.txt', '*.rst'],
        # If you have additional data files in your package, list them here
    },
)