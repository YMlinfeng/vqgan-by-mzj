from setuptools import setup, find_packages

# 读取 README.md 作为项目描述（如果没有 README.md 可以先创建一个）
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='vqgan-by-mzj',  # 包名，确保 PyPI 上没有重名
    version='1.0.0',
    description='Mzj_With_VQGAN',
    long_description=long_description,  # 显示在 PyPI 页面上的详细描述
    long_description_content_type='text/markdown',  # 告诉 PyPI README 是 markdown 格式
    author='Zijie Meng',
    author_email='xiao102851@163.com',
    url='https://ymlinfeng.github.io/',  # GitHub 或项目主页地址

    packages=find_packages(),  # 自动查找所有含 __init__.py 的包

    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # 如果你使用 MIT 协议
        'Operating System :: OS Independent',
    ],

    python_requires='>=3.6',  # 支持的最低 Python 版本
)



'''
packages=find_packages(),
自动查找所有的 Python 包（即包含 __init__.py 的目录）。

例如，如果你的目录结构是：


复制
taming/
  __init__.py
  modules/
    __init__.py
    something.py
则 find_packages() 会找到 taming 和 taming.modules 这两个包。



'''

'''
对比总结
命令	来源	使用条件	场景
pip install .	本地目录	当前目录包含 setup.py	安装本地开发代码，调试项目
pip install <包名>	PyPI（远程）	包必须发布到 PyPI	安装第三方库、正式发布版本


'''