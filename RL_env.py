# engine_env.py
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from scipy.io import loadmat
from DNN import DNNModel

class DualFuelEngineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mat_path='firstDataset.mat'):
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

        # Observations and actions
        self.actions_raw = np.column_stack((self.doi_main, self.soi_pre, self.soi_main, self.doi_h2))
        self.observations_raw = np.column_stack((self.imep, self.nox, self.pm, self.mprr))

        self.action_low, self.action_high = self._get_bounds(self.actions_raw)
        self.obs_low, self.obs_high = self._get_bounds(self.observations_raw)

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        self.state = None
        self.current_step = 0
        self.dataset_size = self.actions_raw.shape[0]

        # Load trained PyTorch model
        self.model = DNNModel(input_size=8, hidden_size_1=31, hidden_size_2=23, output_size=4)
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()

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
        action = np.clip(action, self.action_low, self.action_high)

        # Create input for the model: concatenate obs + action
        input_array = np.concatenate([self.state, action])
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        with torch.no_grad():
            next_obs_tensor = self.model(input_tensor)
        next_obs = next_obs_tensor.numpy()

        imep, nox, pm, mprr = next_obs

        # Reward calculation
        reward = (
            +1.0 * imep
            +1.0 * nox
            +1.0 * pm
            +1.0 * mprr
        )

        done = False
        truncated = False
        self.current_step = (self.current_step + 1) % self.dataset_size
        self.state = next_obs

        return next_obs.astype(np.float32), reward, done, truncated, {}

    def render(self):
        imep, nox, pm, mprr = self.state
        print(f"IMEP={imep:.2f}, NOx={nox:.1f}, PM={pm:.1f}, MPRR={mprr:.2f}")

    def close(self):
        pass
