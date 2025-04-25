## Policy Gradient Methods

This project implements REINFORCE and Soft Actor Critic (SAC) algorithms to make a robot learn how to push an object to the goal position.

### REINFORCE
I used the gym enviroment with Pusher-v5 for faster training. The algorithm calculates a baseline using weighted averages of returns. I tried the algorithm many times but it did not train well without some weight initializations. Using kaiming_normal_ and xavier_normal_ somehow worked and the latest training worked. The smoothened reward plot can be seen here: \
<img src="https://github.com/deniz6221/PolicyGradientMethods/blob/main/REINFORCE/Gym_Env/Reward_Plot.png" alt="REINFORCE_RPLT" style="max-width: 100%; display: inline-block;" width="800px"> \
The model performs well under most cases but there are a few edge cases where it fails. Here are some cases with successful outcomes:
<img src="https://github.com/deniz6221/PolicyGradientMethods/blob/main/REINFORCE/Gym_Env/gifs/R_1.gif" alt="REINFORCE1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PolicyGradientMethods/blob/main/REINFORCE/Gym_Env/gifs/R_2.gif" alt="REINFORCE2" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PolicyGradientMethods/blob/main/REINFORCE/Gym_Env/gifs/R_3.gif" alt="REINFORCE3" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PolicyGradientMethods/blob/main/REINFORCE/Gym_Env/gifs/R_4.gif" alt="REINFORCE4" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
### Soft Actor Critic

I used one of the first SAC algorithms to train the model. The algorithm tries to maximize entropy and reward while trying to minimize the difference between target and evaluated Q values. I kept the entropy temperature alpha constant (0.2),
the better way to do it is to make the algorithm learn alpha but I did not implement it. The training was successful after 5000 episodes. The smoothened reward plot can be seen here: \
<img src="https://github.com/deniz6221/PolicyGradientMethods/blob/main/SoftActorCritic/Reward_Plot.png" alt="SAC_RPLT" style="max-width: 100%; display: inline-block;" width="800px"> \
The model performs good under most cases, here are some harder ones with successful outcomes: \
<img src="https://github.com/deniz6221/PolicyGradientMethods/raw/main/SoftActorCritic/gifs/SAC_1.gif" alt="SAC1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PolicyGradientMethods/raw/main/SoftActorCritic/gifs/SAC_2.gif" alt="SAC1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PolicyGradientMethods/raw/main/SoftActorCritic/gifs/SAC_3.gif" alt="SAC1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">

### How To Use The Models

#### REINFORCE
CD to REINFORCE directory with `cd REINFORCE/Gym_Env` \
To train the REINFORCE model, call the `train()` function from the main.py file \
To train the REINFORCE model, call the `test()` function from the main.py file
#### SAC
CD to SAC directory with `cd SoftActorCritic` \
To test the SAC model, run `python3 test_model.py` \
To train the SAC model, run `python3 homework3.py`
