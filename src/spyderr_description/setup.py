from setuptools import setup
import os
from glob import glob


from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
def package_files(directory, install_base):
    """Return a list of (install_dir, [files]) preserving subdirectories."""
    directory = (PACKAGE_ROOT / directory).resolve()
    data = {}
    for path in directory.rglob('*'):
        if path.is_file():
            rel_parent = path.parent.relative_to(directory)  # '.' or subdir
            install_dir = str(Path(install_base) / rel_parent)
            data.setdefault(install_dir, []).append(str(path.relative_to(PACKAGE_ROOT)))
    return sorted(data.items())

package_name = 'spyderr_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        # Meshes (preserve subdirectories, incl. meshes/link_local/)
        *package_files('meshes', os.path.join('share', package_name, 'meshes')),


        (os.path.join('share', package_name, 'config'), glob('config/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='author',
    maintainer_email='todo@todo.com',
    description='The ' + package_name + ' package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
