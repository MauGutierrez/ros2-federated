# ros2-federated


# Introduction
ROS2-mnist is a project developed to experiment an integration between ROS framework and Tensorflow.
The objective is to have a Federated Learning architecture, in which a list of services will be executed to train a local neural network. 

# Technologies
Project is created with:
* Ubuntu 22.04
* Python 3.10.6
* Robot Operating System 2 (ROS2) Humble Hawksbill distro
* Tensorflow 2.12.0
* Tmux (optional)

# Setup
To run this project, go to the root:
```
$ cd ros2_federated
```
Once there, clean and compile the project:
```
$ . clean.sh
$ . build.sh
```
Now that the project is already compiled, execute the Federated Server with the next command, passing as argument the number of clients that will be working, and the selected dataset:
```
$ ros2 run ros2_federated main 1 mnist
```
On our example, we are specifying only one client and mnist dataset.

To start a client, first open a new termianl and source the ROS project with the next script:
```
$ . init.sh
```

Then, execute the next command and pass as argument the same dataset for the Federated Server:
```
$ ros2 run ros2_federated client_1 mnist
```

If you were to use more clients, don't forget to specify that in the commands to execute the Federated Server and the clients. Also, don't forget to use the same dataset for the Federated Server and the client.

After running the client, local training will begin.

You can change the model hyperparameters on the seetings json file.