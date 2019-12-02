# DDPG Algorithm - Colaboration
---

This work is part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) third assignment, which consists on solving the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. 

![tennis-env](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

## Setup
---

To setup your local Python environment for running Unity environments checkout the instructions on [this Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). On this work we'll use [PyTorch](https://pytorch.org/) to build the networks. On `requirements.txt` you'll also find some other packages required.

## Environment
---

This work does **not** require to install Unity, the environment is already been built, and you can download it from the link below:

* Linux: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OS X: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32 bit): [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64 bit): [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)

Then you must place the environment inside the `env` folder, or update the path on the notebook, if you wish to reproduce the `report.ipynb`.

## Development
---

You should follow `report.ipynb` for the detailed implementation process. The `models` folder holds all the model files that was used, and `utils` folder has the support files, such as the `noise` and `replay buffer` implementations. On the `agent.py` file the **main agent** is implemented, the one responsible for creating and training both the `actor` and `critic` networks.

## Trained model
---

You can use the trained model by loading the parameters from `model_parameters` folder to both the `actor` and `critic` network, and then acting on the environment. 

## Future work
--- 

This work can be improved by testing another noise functions, as well as implementing [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf).
