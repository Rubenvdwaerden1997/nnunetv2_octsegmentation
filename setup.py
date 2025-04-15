from setuptools import setup, find_packages

setup(
    name='nnunetv2_octsegmentation',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    entry_points={'console_scripts': [
        'nnUNet_predict = nnunet.inference.predict:predict_entry_point',
        ],
    }
)