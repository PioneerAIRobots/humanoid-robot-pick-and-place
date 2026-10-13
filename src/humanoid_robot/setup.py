import os
from glob import glob

from setuptools import setup

package_name = 'humanoid_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='khan',
    maintainer_email='platformanalytics23@gmail.com',
    description='Humanoid robot URDF description and keyboard teleoperation for RViz',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keyboard_teleop = humanoid_robot.keyboard_teleop:main',
            'lidar_simulator = humanoid_robot.lidar_simulator:main',
            'navigate_to_goal = humanoid_robot.navigate_to_goal:main',
        ],
    },
)
