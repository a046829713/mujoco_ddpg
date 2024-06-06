import gymnasium as gym
from model import Actor, Critic
import numpy as np
import time
import torch
import torch.optim as optim
import os


# 按下tab即可以讓相機追蹤
class Tester:
    def __init__(self, env_name, actor_path, num_episodes=10):
        self.env = gym.make(env_name, render_mode='human')
        self.num_episodes = num_episodes
        self.actor = Actor(self.env.observation_space.shape[0], 
                           self.env.action_space.shape[0], 
                           self.env.action_space.high[0])
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor.eval()
        self.max_episode_length = 1000  # 根據gym的描述 1000步就會結束

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy().squeeze(0)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def test(self):
        for i_episode in range(self.num_episodes):
            state, episode_return, episode_length, d = self.env.reset(), 0, 0, False
            state = state[0]

            while not (d or (episode_length == self.max_episode_length)):
                self.env.render()
                action = self.get_action(state)  # No noise during testing
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                episode_length += 1
                state = next_state

            print("Test Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length)



if __name__ == "__main__":
    tester = Tester('HalfCheetah-v4', 'models/actor.pth', num_episodes=1)
    tester.test()