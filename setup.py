"""
Setup script.
"""

from setuptools import setup, find_packages


def _read(fpath):
    fcontent = ""
    with open(fpath, "r", encoding="utf8") as file:
        fcontent = file.read()
    return fcontent


setup(
    name="projectUQA",
    version="0.1a0",
    packages=find_packages(),
    install_requires=["docutils", "click"],
    entry_points="""
        [console_scripts]
        uqa=uqa.cli:main
    """,
    # metadata to display on PyPI
    description="Unsupervised Question Answering project CentraleSupéléc / Illuin technologies",
    author="Tronch Boris, Lu Jiahao, Churet Quentin",
    author_email="quentin.churet@student.ecp.fr, boris.tronch@student.ecp.fr, jiahao.lu@student.ecp.fr",
    url="https://github.com/bonoboris/Project-Unsupervised-Question-Answering'",  # project home page, if any
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
