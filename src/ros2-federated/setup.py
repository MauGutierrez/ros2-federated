from setuptools import setup

import os

package_name = 'ros2_federated'
share_dir = os.path.join("share", package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join(share_dir, "config"), ["config/settings.json"]),
        (os.path.join(share_dir, "config"), ["config/params.yaml"]),
        (os.path.join(share_dir, "launch"), ["launch/clients.launch.py"]),
        (os.path.join(share_dir, "launch"), ["launch/server.launch.py"])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mauricio',
    maintainer_email='mauricio@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = ros2_federated.ros2_federated_server:main',
            'client_1 = ros2_federated.client_1:main',
            'client_2 = ros2_federated.client_2:main'
        ],
    },
)
