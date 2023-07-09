from setuptools import find_packages
from setuptools import setup

package_name = 'ros2_mnist'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['settings.json']),
        ('share/' + package_name + '/dataset', 
            ['dataset/' + 'test.csv']),
        ('share/' + package_name + '/dataset', 
            ['dataset/' + 'test.zip']),
        ('share/' + package_name + '/dataset', 
            ['dataset/' + 'train.csv']),
        ('share/' + package_name + '/dataset', 
            ['dataset/' + 'train.zip']),
        ('share/' + package_name + '/dataset', 
            ['dataset/' + 'validation.csv']),
            ('share/' + package_name + '/dataset', 
            ['dataset/' + 'validation.zip']),
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
            'main = ros2_mnist.ros2_federated_server:main',
            'client_1 = ros2_mnist.client_1:main',
            'client_2 = ros2_mnist.client_2:main'
        ],
    },
)
