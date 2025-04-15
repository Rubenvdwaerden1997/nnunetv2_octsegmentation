from setuptools import setup, find_packages

setup(
    name='nnunetv2',
    version='0.1',
    packages=find_packages(include=["nnunetv2", "nnunetv2.*"]),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'nnUNet_predict = nnunetv2.inference.predict_from_raw_data:predict_entry_point',
        ],
    }
)