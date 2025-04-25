import gymnasium as gym
import numpy as np
from agent import Agent
import json
import torch

def custom_reward(ee_pos, obj_pos, goal_pos, prev_obj_pos, is_terminal):
         
        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # distance-based rewards
        r_ee_to_obj = -0.1 * d_ee_to_obj  # getting closer to object
        r_obj_to_goal = -0.2 * d_obj_to_goal  # moving object to goal

        # direction bonus
        obj_movement = obj_pos - prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        r_direction = 0.5 * max(0, np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal))
        if np.linalg.norm(obj_movement) < 1e-6:  # Avoid division by zero
            r_direction = 0.0


        # terminal bonus
        r_terminal = 10.0 if is_terminal else 0.0

        r_step = -0.1  # penalty for each step

        return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step


def train():
    prev_obj_pos = None
    env = gym.make("Pusher-v5", max_episode_steps=100)

    action_dim = env.action_space.shape[0]

    observation_dim = 9

    N_EPISODE = 40_000

    agent = Agent(act_dim=action_dim, obs_dim=observation_dim)

    ee_start_idx = 14

    rews = []
    for episode in range(1, N_EPISODE + 1):
        done = False
        observation, _ = env.reset()
        
        x_ee, y_ee, z_ee = observation[ee_start_idx : ee_start_idx + 3]
        x_obj, y_obj, z_obj = observation[ee_start_idx +3 : ee_start_idx + 6]
        x_goal, y_goal, z_goal = observation[ee_start_idx + 6 : ee_start_idx + 9]
        
        state = [x_ee, y_ee, z_ee, x_obj, y_obj, z_obj, x_goal, y_goal, z_goal]
        state = torch.tensor(state, dtype=torch.float32)
        prev_obj_pos = np.array([x_obj, y_obj, z_obj])
        cumulative_reward = 0


        while not done:
            action = agent.decide_action(state)
            observation, _, done, truncated, _ = env.step(action=action)

            next_x_ee, next_y_ee, next_z_ee = observation[ee_start_idx : ee_start_idx + 3]
            next_x_obj, next_y_obj, next_z_obj = observation[ee_start_idx +3 : ee_start_idx + 6]
            next_x_goal, next_y_goal, next_z_goal = observation[ee_start_idx + 6 : ee_start_idx + 9]
            next_state = [next_x_ee, next_y_ee, next_z_ee, next_x_obj, next_y_obj, next_z_obj, next_x_goal, next_y_goal, next_z_goal]

            reward = custom_reward(
                ee_pos=np.array([next_x_ee, next_y_ee, next_z_ee]),
                obj_pos=np.array([next_x_obj, next_y_obj, next_z_obj]),
                goal_pos=np.array([next_x_goal, next_y_goal, next_z_goal]),
                prev_obj_pos=prev_obj_pos,
                is_terminal=done
            ) 


            next_state = torch.tensor(next_state, dtype=torch.float32)

            done = done or truncated

            
            prev_obj_pos = np.array([next_x_obj, next_y_obj, next_z_obj])
            cumulative_reward += reward
            agent.add_reward(reward)
            state = next_state

        rews.append(float(cumulative_reward))
        agent.update_model()          
        print(f"Episode: {episode}, Cumulative Reward: {cumulative_reward:.2f}")

        if episode % 1000 == 0:
            agent.save_checkpoint(f"checkpoints/vpg_model_{episode}.pth")
            with open(f"checkpoints/rewards_{episode}.json", "w") as f:
                json.dump(rews, f)

    
    # Save the model
    torch.save(agent.model.state_dict(), "checkpoints/vpg_model.pth")

    #Save rewards
    with open("checkpoints/rewards.json", "w") as f:
        json.dump(rews, f)

def test():
    env = gym.make("Pusher-v5", max_episode_steps=500, render_mode="human")

    action_dim = env.action_space.shape[0]

    observation_dim = 9

    N_EPISODE = 40_000

    agent = Agent(act_dim=action_dim, obs_dim=observation_dim)
    agent.load_model("checkpoints/vpg_model.pth")
    for i in range(15):
        ee_start_idx = 14

        done = False
        observation, _ = env.reset()
        
        x_ee, y_ee, z_ee = observation[ee_start_idx : ee_start_idx + 3]
        x_obj, y_obj, z_obj = observation[ee_start_idx +3 : ee_start_idx + 6]
        x_goal, y_goal, z_goal = observation[ee_start_idx + 6 : ee_start_idx + 9]
        
        state = [x_ee, y_ee, z_ee, x_obj, y_obj, z_obj, x_goal, y_goal, z_goal]
        state = torch.tensor(state, dtype=torch.float32)

        while not done:
            action = agent.decide_action_fixed(state)
            observation, _, done, truncated, _ = env.step(action=action)

            next_x_ee, next_y_ee, next_z_ee = observation[ee_start_idx : ee_start_idx + 3]
            next_x_obj, next_y_obj, next_z_obj = observation[ee_start_idx +3 : ee_start_idx + 6]
            next_x_goal, next_y_goal, next_z_goal = observation[ee_start_idx + 6 : ee_start_idx + 9]
            next_state = [next_x_ee, next_y_ee, next_z_ee, next_x_obj, next_y_obj, next_z_obj, next_x_goal, next_y_goal, next_z_goal]


            next_state = torch.tensor(next_state, dtype=torch.float32)

            done = done or truncated

            state = next_state


if __name__ == "__main__":
    test()