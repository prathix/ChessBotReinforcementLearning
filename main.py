import gym
from pettingzoo.classic import chess_v6
from gym import spaces
import numpy as np

# Custom wrapper class to convert PettingZoo to Gym
class PettingZooToGymEnv(gym.Env):
    def __init__(self, env):
        super(PettingZooToGymEnv, self).__init__()
        self.env = env
        self.agents = env.possible_agents
        self.current_agent = self.agents[0]  # Start with the first agent (you can modify logic here)
        
        # Define action space and observation space for the current agent
        action_space = env.action_space(self.current_agent)
        observation_space = env.observation_space(self.current_agent)
        
        # Ensure that action_space is either Discrete or Box
        if isinstance(action_space, spaces.Discrete):
            self.action_space = spaces.Discrete(action_space.n)
        elif isinstance(action_space, spaces.Box):
            self.action_space = spaces.Box(low=action_space.low, high=action_space.high, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action_space type: {type(action_space)}")
        
        # Ensure that observation_space is either Box or Discrete
        if isinstance(observation_space, spaces.Box):
            self.observation_space = spaces.Box(low=observation_space.low, high=observation_space.high, dtype=np.float32)
        elif isinstance(observation_space, spaces.Discrete):
            self.observation_space = spaces.Discrete(observation_space.n)
        else:
            raise ValueError(f"Unsupported observation_space type: {type(observation_space)}")
    
    def reset(self):
        self.env.reset()  # Reset the PettingZoo environment
        self.current_agent = self.agents[0]  # Reset to first agent (you can change logic)
        return self.env.observation(self.current_agent)  # Return the observation for the current agent
    
    def step(self, action):
        # Take the action in the environment and observe the result
        self.env.step({self.current_agent: action})
        
        # Get the next state, reward, done, and additional info
        obs = self.env.observation(self.current_agent)
        reward = self.env.rewards[self.current_agent]
        done = self.env.dones[self.current_agent]
        info = self.env.infos[self.current_agent]
        
        return obs, reward, done, info
    
    def render(self, mode="human"):
        # Optionally, implement rendering here
        self.env.render()

    def close(self):
        self.env.close()

# Load the PettingZoo chess environment
env = chess_v6.env()
env.metadata["is_parallelizable"] = True
env.reset()

# Convert the PettingZoo environment to a Gym-compatible one
gym_env = PettingZooToGymEnv(env)

# Now gym_env is a Gym-compatible environment
# You can use it with Stable-Baselines3 or any Gym-compatible algorithms

# Train the model
from stable_baselines3 import PPO
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("chess_ai")
