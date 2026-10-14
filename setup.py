"""Package the FSTA-Det MMDetection fork."""

from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
requirements = [
    line.strip() for line in (root / 'requirements.txt').read_text().splitlines()
    if line.strip() and not line.startswith('#')
]

setup(
    name='fsta-det',
    version='0.1.0',
    description='Frequency enhancement and cross-window aggregation for tiny object detection',
    long_description=(root / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/jiangjoker99-ctrl/FSTA-Det',
    packages=find_packages(include=['mmdet', 'mmdet.*']),
    python_requires='>=3.8',
    install_requires=requirements,
    license='Apache-2.0',
    zip_safe=False)
