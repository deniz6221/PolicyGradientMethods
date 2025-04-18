## Policy Gradient Methods

This project implements REINFORCE and Soft Actor Critic (SAC) algorithms to make a robot learn how to push an object to the goal position.

### REINFORCE
My goal was to train the model for 10000 episodes, however the rewwards never improved so I killed the training after 7000 episodes.
The reward/iteration plot after 7002 episodes can be found in REINFORCE/Reward_Plot_New.png
The trained model can be found in REINFORCE/checkpoints/checkpoint_new_7002.pth

### Soft Actor Critic

I used one of the first SAC algorithms to train the model. The algorithm tries to maximize entropy and reward while trying to minimize the difference between target and evaluated Q values. I kept the entropy temperature alpha constant (0.2),
the better way to do it is to make the algorithm learn alpha but I did not implement it. The training was successful after 5000 episodes. The reward plot can be found at SoftActorCritic/Reward_Plot.png \
The model performs good under most cases, here are some harder ones with successful outcomes: \
<img src="https://github.com/deniz6221/PolicyGradientMethods/raw/main/SoftActorCritic/gifs/SAC_1.gif" alt="SAC1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PolicyGradientMethods/raw/main/SoftActorCritic/gifs/SAC_2.gif" alt="SAC1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PolicyGradientMethods/raw/main/SoftActorCritic/gifs/SAC_3.gif" alt="SAC1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">

### How To Use The Models

To test the SAC model, run `python3 SoftActorCritic/test_model.py` \
To train the SAC model, run `python3 SoftActorCritic/homework3.py`
