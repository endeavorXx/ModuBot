from setuptools import setup
import os
from glob import glob

package_name = 'my_nao_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Include the launch directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        
        # Include the worlds directory (for .wbt files)
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.wbt')),
        
        # Include the resource directory (for .urdf files)
        (os.path.join('share', package_name, 'resource'), glob('resource/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vashu Chauhan',
    maintainer_email='vashu22606@iiitd.ac.in',
    description='Nao Controller Driver for Webots',
    license='MIT License',
    tests_require=['pytest'],
    
    entry_points={
        'console_scripts': [
            # This registers the command 'test_actions' to run the 'main' function
            # inside the file 'test_nao_actions.py'
            'test_actions = my_nao_controller.test_nao_actions:main',
            'nao_brain = my_nao_controller.nao_brain:main',
        ],
    },
)