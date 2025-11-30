from setuptools import setup
import os
from glob import glob

package_name = 'op2_controller'

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
    description='Robotis OP2 Controller Driver for Webots',
    license='MIT License',
    
    entry_points={
        'console_scripts': [
            'op2_brain = op2_controller.op2_brain:main',
            'op2_driver = op2_controller.op2_driver:main',
            'test_actions = op2_controller.test_op2_actions:main',
        ],
    },
)
