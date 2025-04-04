## Policy Gradient Methods

This project implements REINFORCE and Soft Actor Critic (SAC) algorithms to make a robot learn how to push an object to the goal position.

### REINFORCE
My goal was to train the model for 10000 episodes, however the rewards being disproportional caused exploding gradients and that made training impossible with high learning rates.
I tried gradient clipping and other methods to keep the gradient in check but none of them worked, only solution was to lower the learning rate. I tried several learning rates by halfing them when they fail, the best one was 5e-6.
However it also failed after 5000+ episodes, I didn't bother to lower the learning rate more since the learning became quite inefficient with such low rates. The reward/iteration plot after 4998 episodes can be found in REINFORCE/Reward_Plot.png
The trained model can be found in REINFORCE/checkpoints/checkpoint_4995.pth

### Soft Actor Critic

I used one of the first SAC algorithms to train the model. The algorithm tries to maximize entropy and reward while trying to minimize the difference between target and evaluated Q values. I kept the entropy temperature alpha constant (0.2),
the better way to do it is to make the algorithm learn alpha but I did not implement it. The training once again crashed but at 6000+ episodes due to exploding gradients. The plot can be found in SoftActorCritic/Reward_Plot.png
The trained model can be found in SoftActorCritic/checkpoints/checkpoint_6000.pth

### How To Use The Models
The checkpoint files contain both the optimizer and model state dictionaries. To use the models, one must load the file into memory as a dictionary and get the models from the dictionary.
