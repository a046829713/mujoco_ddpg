import gymnasium as gym
from model import Actor, Critic
import numpy as np
import time
import torch
import torch.optim as optim
import os

### The experience replay memory ###
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


class DDPG():
    def __init__(self) -> None:
        self.hyperparameter()

        self.env = gym.make('HalfCheetah-v4', render_mode='rgb_array')

        # get size of state space and action space
        self.state_size = self.env.observation_space.shape[0]  # 17

        # Box(-1, 1, (6,), float32)
        self.action_size = self.env.action_space.shape[0]



        # Experience replay memory
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.state_size, act_dim=self.action_size, size=self.replay_size)

        # Maximum value of action
        # Assumes both low and high values are the same
        # Assumes all actions have the same bounds
        # May NOT be the case for all environments
        self.action_max = self.env.action_space.high[0]
        self.create_model()
        self.train()

    def create_model(self):
        self.actor = Actor(self.state_size, self.action_size, self.action_max)
        self.target_actor = Actor(
            self.state_size, self.action_size, self.action_max)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_size, self.action_size)
        self.target_critic = Critic(self.state_size, self.action_size)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.q_lr)

    def hyperparameter(self):
        self.replay_size = 100000
        self.num_train_episodes = 100
        self.max_episode_length = 1000  # 根據gym的描述 1000步就會結束
        self.action_noise = 0.1  # 動作噪音
        self.start_steps = 10000
        self.batch_size = 100
        self.discount_factor = 0.99
        self.actor_lr = 1e-3
        self.q_lr = 1e-3
        self.target_update_freq = 100  # 定義多久進行一次軟更新，例如每100次訓練步驟
        self.update_count = 0  # 計數器追踪更新次數

    def get_action(self, s, noise_scale):
        # Convert state to PyTorch tensor
        state_tensor = torch.from_numpy(
            s).float().unsqueeze(0)  # Add batch dimension

        # Get action from the actor network
        a = self.actor(state_tensor)

        # Convert action tensor to numpy array and squeeze the batch dimension
        a = a.detach().numpy().squeeze(0)

        # Add noise for exploration
        noise = noise_scale * np.random.randn(self.action_size)
        a += noise

        # Clip the actions to be within the allowed range
        a = np.clip(a, -self.action_max, self.action_max)
        return a

    def update(self, batch_size):
        # 從回放緩衝區抽樣一批體驗
        batch = self.replay_buffer.sample_batch(batch_size)
        s, a, r, s2, d = batch['s'], batch['a'], batch['r'], batch['s2'], batch['d']

        s = torch.from_numpy(s)
        s2 = torch.from_numpy(s2)
        d = torch.from_numpy(d)
        r = torch.from_numpy(r)
        a = torch.from_numpy(a)

        # 使用下一狀態和目標批評者網絡計算目標Q值
        with torch.no_grad():
            next_actions = self.target_actor(s2)
            target_q_values = self.target_critic(s2, next_actions)
            target_q = r.unsqueeze(1) + self.discount_factor * \
                target_q_values * (1 - d).unsqueeze(1)

        # 更新批評者網絡
        critic_q_values = self.critic(s, a)
        critic_loss = torch.nn.functional.mse_loss(critic_q_values, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新演員網絡
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, local_model, target_model, tau):
        # 軟更新模型參數
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self):
        # Main loop: play episode and train
        returns = []
        q_losses = []
        mu_losses = []
        num_steps = 0

        for i_episode in range(self.num_train_episodes):
            # reset env
            state, episode_return, episode_length, d = self.env.reset(), 0, 0, False
            state = state[0]

            while not (d or (episode_length == self.max_episode_length)):
                # For the first `start_steps` steps, use randomly sampled actions
                # in order to encourage exploration.
                if num_steps > self.start_steps:
                    action = self.get_action(state, self.action_noise)
                else:
                    action = self.env.action_space.sample()

                # Keep track of the number of steps done
                num_steps += 1
                if num_steps == self.start_steps:
                    print("USING AGENT ACTIONS NOW")

                # # Step the env
                next_state, reward, terminated, truncated, info = self.env.step(
                    action)
                episode_return += reward
                episode_length += 1

                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d_store = False if episode_length == self.max_episode_length else terminated

                # Store experience to replay buffer
                self.replay_buffer.store(
                    state, action, reward, next_state, d_store)

                # Assign next state to be the current state on the next round
                state = next_state

                
            # Perform the updates
            for _ in range(episode_length):
                self.update(batch_size=self.batch_size)
                self.update_count += 1

                if self.update_count % self.target_update_freq == 0:  # 每隔一定次數進行一次軟更新
                    self.soft_update(self.critic, self.target_critic, tau=0.995)
                    self.soft_update(self.actor, self.target_actor, tau=0.995)

            print("Episode:", i_episode + 1, "Return:",
              episode_return, 'episode_length:', episode_length)
            returns.append(episode_return)
        
        
        # Save the models after training
        self.save_model()
        print("Training finished. Models saved successfully.")
    
    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.actor.state_dict(), 'models/actor.pth')
        torch.save(self.critic.state_dict(), 'models/critic.pth')
        print("Models saved successfully.")



if __name__ == "__main__":
    # 開始訓練
    DDPG()
