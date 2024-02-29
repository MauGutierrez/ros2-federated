from setuptools import find_packages, setup

import os

package_name = 'ros2_rl_agents'
share_dir = os.path.join("share", package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join(share_dir, "config"), ["config/settings.json"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mauricio',
    maintainer_email='gerardo.gutierrezq@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent_1 = ros2_rl_agents.ros2_rl_agents:main',
        ],
    },
)
