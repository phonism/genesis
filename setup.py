from setuptools import setup, find_packages

setup(
    name='Genesis',  # 包名称
    version='0.1.0',  # 版本
    author='',  # 作者
    author_email='',  # 作者邮箱
    description='Gensis is a lightweight deep learning framework written from scratch in Python, with Triton as its backend for high-performance computing.',  # 简短描述
    long_description=open('README.md').read(),  # 长描述，通常从README文件中读取
    long_description_content_type='text/markdown',  # 长描述的格式
    url='https://github.com/phonism/genesis',  # 项目主页
    packages=find_packages(),  # 自动找到项目中的所有包
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # 许可证
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Python版本要求
    install_requires=[
        "numpy",
        "torch"
    ],
    entry_points={
        'console_scripts': [
            # 'your_command=your_package.module:function',
        ],
    },
    include_package_data=True,  # 包含包内的数据文件
    package_data={
        # '': ['*.txt', '*.rst'],
        # 如果你的包中有额外的数据文件，需要在这里列出
    },
)
