# Masterthesis

This Repo contains everything related to my master thesis. 

## Masterthesis abstract 

* Title: Autonomous Modeling of a 3D Environment with
Drones
* First reviewer: Prof. (FH) PD Dr. Mario Döller
* Second reviewer:  Sebastian Danninger, MA

Within the context of autonomous exploration and modeling of a 3D environment with drones, this work targets two distinct research questions. Most
existing autonomous exploration frameworks divide the exploration task into
three subproblems: localization, mapping, and path planning [1]. The location
and mapping problem, which refers to the task of generating a local map of
the environment while localizing the drone within it, is addressed by several
already existing SLAM algorithms. The first examined research question to
answer is what the best suited open-source visual SLAM algorithm for the
exploration task is. Three algorithms were evaluated using predefined criteria
addressing trajectory accuracy, point cloud accuracy and computation time.
The results showed, that ORB SLAM outperforms other algorithms. The second research task was to develop a framework, that enables users to test and
implement fully autonomous exploration systems within a virtual environment. The framework was developed in ROS and provides the possibility to
navigate a drone within a simulation while the ORB SLAM algorithm is applied on the drone’s camera output. With the help of suited transformations
on the output of the ORB SLAM algorithm, ground truth data is then streamed
within the framework and enables users to apply and test flight path planning
algorithms in order to complete the autonomous exploration task.



## Repo Structure

#### /evaluation 
Contains everything related to the first chapter, the evaluation

#### /thesis
Contains everything related to writing the actual thesis including latex setup

#### /ros_packages
Contains everything related to the proposed framework in section two. 
