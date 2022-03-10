from setuptools import setup, find_packages

setup(name="eznukutils",
      version="0.1.1",
      python_requires='>=3.8',
      description="Utility modules",
      license="MIT",
      packages=find_packages(),
      install_requires=["numpy"],
      )