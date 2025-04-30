import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import cv2

class CNNPolicy(BaseFeaturesExtractor):
    """CNN architecture for processing game frames"""
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        return self.linear(self.cnn(observations))

class TransformerPolicy(BaseFeaturesExtractor):
    """Transformer architecture for processing game frames"""
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # Patch embedding
        self.patch_size = 8
        self.patch_embed = nn.Conv2d(
            n_input_channels, 
            features_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Calculate number of patches
        h, w = observation_space.shape[1:]
        n_patches = (h // self.patch_size) * (w // self.patch_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, features_dim))
        
    def forward(self, observations):
        # Patch embedding
        x = self.patch_embed(observations)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, C)
        return x

class HybridCNNTransformerPolicy(BaseFeaturesExtractor):
    """Hybrid CNN-Transformer architecture"""
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            cnn_output = self.cnn(torch.zeros(1, *observation_space.shape))
            _, c, h, w = cnn_output.shape
            n_patches = h * w
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )
        
        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(c * h * w, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        # CNN features
        x = self.cnn(observations)
        b, c, h, w = x.shape
        
        # Reshape for transformer
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Reshape and apply final layers
        x = x.transpose(1, 2).flatten(1)  # (B, C*H*W)
        x = self.fc(x)
        return x

class DoomEnv(gym.Env):
    """Custom Doom environment"""
    def __init__(self, config_path="basic.cfg", frame_skip=4):
        super().__init__()
        
        # Initialize VizDoom
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.init()
        self.frame_skip = frame_skip
        
        # Define action space (based on available buttons)
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())
        
        # Define observation space (84x84 grayscale image)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, 84, 84),  # 4 stacked frames
            dtype=np.uint8
        )
        
        # Frame buffer for frame stacking
        self.frames = np.zeros((4, 84, 84), dtype=np.uint8)
        
        # Random number generator
        self.np_random = None
        
    def seed(self, seed=None):
        """Set the seed for this env's random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # Set seed for VizDoom's random number generator
        self.game.set_seed(seed)
        return [seed]
        
    def _preprocess_frame(self, frame):
        """Convert frame to grayscale and resize"""
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame
        
    def _update_frames(self, new_frame):
        """Update frame stack with new frame"""
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = new_frame
        
    def step(self, action):
        """Execute action and return new observation"""
        # Convert action to VizDoom format
        reward = self.game.make_action([action == i for i in range(self.action_space.n)], self.frame_skip)
        done = self.game.is_episode_finished()
        
        if not done:
            # Get state and preprocess
            state = self.game.get_state().screen_buffer
            frame = self._preprocess_frame(state)
            self._update_frames(frame)
        
        info = {}
        return self.frames, reward, done, False, info  # False for truncated, empty dict for info
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        frame = self._preprocess_frame(state)
        self.frames.fill(0)
        self._update_frames(frame)
        
        return self.frames, {}  # Empty dict for info
        
    def render(self, mode="rgb_array"):
        """Render current game state"""
        if mode == "rgb_array" and not self.game.is_episode_finished():
            return self.game.get_state().screen_buffer.transpose(1, 2, 0)
        return np.zeros((240, 320, 3), dtype=np.uint8)
        
    def close(self):
        """Clean up resources"""
        if self.game is not None:
            self.game.close()
            self.game = None

def create_doom_env(config_path, frame_skip=4):
    """Helper function to create Doom environment"""
    return DoomEnv(config_path, frame_skip)