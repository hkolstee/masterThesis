from setuptools import setup

setup(
    name="meeting_env",
    version="0.0.1",
    install_requires=["gymnasium==0.29.0", "pettingzoo==1.24.3"],
)

# install locally: pip3 install -e .
# when in this folder