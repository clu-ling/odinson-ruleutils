from setuptools import setup
import os

from odinson.ruleutils.info import info

# use requirements.txt as deps list
with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

# get readme
with open("README.md", "r") as f:
    readme = f.read()

test_deps = required + ["green>=2.5.0", "coverage", "mypy"]
# NOTE: <packagename> @ allows installation of git-based URLs
dev_deps = test_deps + [
    "black",
    "mkdocs==1.2.1",
    "jinja2==3.0.3",
    # "portray @ git+git://github.com/myedibleenso/portray.git@issue/83",
    # "portray @ git+git://github.com/myedibleenso/portray.git@avoid-regressions",
    # "mkapi==1.0.14",
    "pdoc3==0.10.0",
    "mkdocs-git-snippet==0.1.1",
    "mkdocs-git-revision-date-localized-plugin==0.9.2",
    "mkdocs-git-authors-plugin==0.3.3",
    "mkdocs-rtd-dropdown==1.0.2",
    "pre-commit==2.13.0",
]

setup(
    name="odinson-ruleutils",
    python_requires=">=3.8",
    packages=["odinson.ruleutils"],
    version=info.version,
    keywords=["nlp"],
    description=info.description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=info.repo,
    download_url=info.download_url,
    author=" and ".join(info.authors),
    author_email=info.contact,
    license=info.license,
    # see https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html
    scripts=[os.path.join("bin", "example-script")],
    install_requires=required,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
    tests_require=test_deps,
    extras_require={"test": test_deps, "all": dev_deps},
    include_package_data=True,
    zip_safe=False,
)
