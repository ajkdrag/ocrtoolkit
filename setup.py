import os
import re

from setuptools import find_packages, setup

regexp = re.compile(r".*__version__ = [\'\"](.*?)[\'\"]", re.S)

base_package = "ocrtoolkit"
base_path = os.path.dirname(__file__)

init_file = os.path.join(base_path, "src", "ocrtoolkit", "__init__.py")
with open(init_file, "r") as f:
    module_content = f.read()

    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError("Cannot find __version__ in {}".format(init_file))

with open("README.rst", "r") as f:
    readme = f.read()

with open("CHANGELOG.rst", "r") as f:
    changes = f.read()


def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    with open(filename, "r") as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


requirements = parse_requirements("requirements.txt")


if __name__ == "__main__":
    setup(
        name="ocrtoolkit",
        description="Parse bank cheques",
        long_description="\n\n".join([readme, changes]),
        license="Apache Software License 2.0",
        url="https://github.com/ajkdrag/ocrtoolkit",
        version=version,
        author="ajkdrag",
        author_email="",
        maintainer="ajkdrag",
        maintainer_email="",
        python_requires="==3.8.*",
        install_requires=requirements,
        extras_require={
            "ultralytics": ["ultralytics==8.1.11"],
            "paddle": ["paddleocr==2.7.0.3", "paddlepaddle-gpu==2.6.0"],
            "doctr": ["python-doctr[torch]==0.7.0"],
            "all": [
                "ultralytics==8.1.11",
                "python-doctr[torch]==0.7.0",
                "paddleocr==2.7.0.3",
                "paddlepaddle-gpu==2.6.0",
            ],
        },
        keywords=["ocrtoolkit"],
        package_dir={"": "src"},
        packages=find_packages("src"),
        zip_safe=False,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
    )
