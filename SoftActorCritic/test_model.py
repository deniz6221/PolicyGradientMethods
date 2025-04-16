from agent import Agent
from homework3 import Hw3Env
import torch


if __name__ == "__main__":

    env = Hw3Env(render_mode="gui")
    agent = Agent()
    agent.load_checkpoint("model.pth")
    cumulative_reward = 0

    env.reset()
    state = env.high_level_state()
    state = torch.tensor(state, dtype=torch.float32)
    done = False


    while not done:
        action = agent.decide_action(state)
        next_state, reward, is_terminal, is_truncated = env.step(action)
        cumulative_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = is_terminal or is_truncated
        agent.replay_buffer.append((state.clone(), action.clone(), reward.clone(), next_state.clone(), done))
        
        state = next_state
        
        

    print(f"Reward={cumulative_reward}")

    