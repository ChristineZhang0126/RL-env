import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from scipy.io import loadmat
from DNN import DNNModel

class DualFuelEngineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mat_path='/home/bob-koch-lab/RL_training/firstDataset.mat'):
        super().__init__()

        # === Load dataset ===
        data = loadmat(mat_path)
        signal = data['Data'][0][0]['signal']

        # Extract signals
        self.imep = signal[0][0].flatten()
        self.nox = signal[1][0].flatten()
        self.pm = signal[2][0].flatten()
        self.mprr = signal[3][0].flatten()

        self.doi_main = signal[4][0].flatten()
        self.soi_pre = signal[5][0].flatten()
        self.soi_main = signal[6][0].flatten()
        self.doi_h2 = signal[7][0].flatten()

        # Split actions and observations
        self.actions_raw = np.column_stack((self.doi_main, self.soi_pre, self.soi_main, self.doi_h2))
        self.observations_raw = np.column_stack((self.imep, self.nox, self.pm, self.mprr))

        # Compute bounds
        self.action_low, self.action_high = self._get_bounds(self.actions_raw)
        self.obs_low, self.obs_high = self._get_bounds(self.observations_raw)

        # Define Gym spaces
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        # Initialize state
        self.state = None
        self.current_step = 0

        # Dataset as transition model (for now)
        self.dataset_size = self.actions_raw.shape[0]

    def _get_bounds(self, data, margin_ratio=0.05):
        min_vals = np.nanmin(data, axis=0)
        max_vals = np.nanmax(data, axis=0)
        margin = (max_vals - min_vals) * margin_ratio
        return min_vals - margin, max_vals + margin

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.dataset_size - 1)
        self.state = self.observations_raw[self.current_step]
        return self.state.astype(np.float32), {}

    def step(self, action):
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)

        # For now: next obs is from dataset (surrogate)
        # TODO: Replace this with your trained PyTorch model
        next_step = (self.current_step + 1) % self.dataset_size
        next_obs = self.observations_raw[next_step]

        imep, nox, pm, mprr = next_obs

        # === Reward function ===
        reward = (
            +1.0 * imep
            - 0.01 * nox
            - 0.05 * pm
            - 0.5 * max(mprr - 15, 0)  # Penalize high MPRR
        )

        done = False
        truncated = False
        self.current_step = next_step
        self.state = next_obs

        return next_obs.astype(np.float32), reward, done, truncated, {}

    def render(self):
        imep, nox, pm, mprr = self.state
        print(f"IMEP={imep:.2f}, NOx={nox:.1f}, PM={pm:.1f}, MPRR={mprr:.2f}")

    def close(self):
        pass
