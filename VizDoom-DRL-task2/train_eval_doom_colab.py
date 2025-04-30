import os
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
from datetime import datetime
import json
import gc
from doom_agents import (
    CNNPolicy, 
    TransformerPolicy, 
    HybridCNNTransformerPolicy, 
    create_doom_env
)

class ExperimentManager:
    def __init__(
        self,
        config_path="/content/gdrive/MyDrive/ML4_vizdoom/configs/basic.cfg",
        n_envs=2,  # Reduced for Colab memory constraints
        n_seeds=3,  # Minimum 3 seeds as per requirements
        total_timesteps=1_000_000,  # Increased for better learning
        eval_episodes=10,
        frame_skip=4
    ):
        # Try to mount Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/gdrive')
            self.base_dir = "/content/gdrive/MyDrive/ML4_vizdoom"
        except ImportError:
            self.base_dir = "."

        self.config_path = config_path
        self.n_envs = n_envs
        self.n_seeds = n_seeds
        self.total_timesteps = total_timesteps
        self.eval_episodes = eval_episodes
        self.frame_skip = frame_skip
        
        # Training hyperparameters optimized for Colab
        self.learning_rate = 3e-4
        self.n_steps = 512  # Reduced for less memory usage
        self.batch_size = 64  # Increased for better stability
        self.n_epochs = 4
        self.clip_range = 0.2
        
        # Setup logging directories
        for dir_name in ['logs', 'models', 'videos']:
            path = os.path.join(self.base_dir, dir_name)
            os.makedirs(path, exist_ok=True)
        
        # Available policy architectures
        self.policies = {
            "cnn": CNNPolicy,
            "transformer": TransformerPolicy,
            "hybrid": HybridCNNTransformerPolicy,
        }
        
        # Set device and memory settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
        
    def make_env(self, seed=None, capture_video=False):
        """Create vectorized environment with error handling"""
        def make_env_fn():
            def _init():
                try:
                    env = create_doom_env(self.config_path, self.frame_skip)
                    if seed is not None:
                        env.seed(seed)
                        env.action_space.seed(seed)
                    env = Monitor(env, os.path.join(self.base_dir, "logs", f"seed_{seed}" if seed is not None else "tmp"))
                    return env
                except Exception as e:
                    print(f"Error creating environment: {str(e)}")
                    return None
            return _init
            
        try:
            envs = [make_env_fn() for _ in range(self.n_envs)]
            env = DummyVecEnv([env for env in envs if env is not None])
            
            if capture_video:
                env = VecVideoRecorder(
                    env,
                    os.path.join(self.base_dir, "videos"),
                    record_video_trigger=lambda step: step % 10000 == 0,
                    video_length=200
                )
                
            return env
        except Exception as e:
            print(f"Error creating vectorized environment: {str(e)}")
            return None
        
    def create_agent(self, policy_type, env):
        """Create PPO agent with memory management"""
        try:
            policy_class = self.policies[policy_type]
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            policy_kwargs = dict(
                features_extractor_class=policy_class,
                features_extractor_kwargs=dict(features_dim=256),
            )
            
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                clip_range=self.clip_range,
                policy_kwargs=policy_kwargs,
                device=self.device,
                verbose=1
            )
            
            return model
        except Exception as e:
            print(f"Error creating agent: {str(e)}")
            return None
        
    def evaluate_agent(self, model, env):
        """Evaluate agent with memory management"""
        episode_rewards = []
        episode_lengths = []
        episode_times = []
        
        try:
            obs = env.reset()
            
            for _ in range(self.eval_episodes):
                done = False
                total_reward = 0
                steps = 0
                start_time = datetime.now()
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward[0]
                    steps += 1
                    if done[0]:
                        obs = env.reset()
                        break
                        
                episode_time = (datetime.now() - start_time).total_seconds()
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                episode_times.append(episode_time)
                
                # Clear memory after each episode
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            'mean_episode_time': float(np.mean(episode_times)),
            'std_episode_time': float(np.std(episode_times))
        }
        
    def train_and_evaluate(self):
        """Train and evaluate with memory management"""
        results = {}
        
        for policy_type in self.policies:
            policy_results = []
            
            for seed in range(self.n_seeds):
                print(f"\nTraining {policy_type} policy (Seed {seed + 1}/{self.n_seeds})")
                
                # Initialize wandb
                try:
                    run = wandb.init(
                        project="vizdoom-rl",
                        name=f"{policy_type}-seed{seed}",
                        config={
                            "policy": policy_type,
                            "seed": seed,
                            "total_timesteps": self.total_timesteps,
                            "n_envs": self.n_envs,
                            "learning_rate": self.learning_rate,
                            "n_steps": self.n_steps,
                            "batch_size": self.batch_size,
                            "n_epochs": self.n_epochs,
                            "clip_range": self.clip_range,
                            "device": str(self.device)
                        },
                        reinit=True
                    )
                except Exception as e:
                    print(f"Error initializing wandb: {str(e)}")
                    continue
                
                # Set random seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                # Create environment and model
                env = self.make_env(seed=seed, capture_video=True)
                if env is None:
                    continue
                    
                model = self.create_agent(policy_type, env)
                if model is None:
                    continue
                
                # Train model
                start_time = time.time()
                try:
                    model.learn(
                        total_timesteps=self.total_timesteps,
                        progress_bar=True
                    )
                except Exception as e:
                    print(f"Training failed: {str(e)}")
                    continue
                    
                train_time = time.time() - start_time
                
                # Save model
                try:
                    save_path = os.path.join(self.base_dir, "models", f"{policy_type}_seed{seed}")
                    model.save(save_path)
                except Exception as e:
                    print(f"Failed to save model: {str(e)}")
                
                # Create evaluation environment
                eval_env = self.make_env(seed=seed+100)
                if eval_env is None:
                    continue
                
                # Evaluate model
                eval_results = self.evaluate_agent(model, eval_env)
                if eval_results is not None:
                    eval_results['train_time'] = train_time
                    policy_results.append(eval_results)
                    
                    # Log evaluation metrics
                    if run is not None:
                        run.log(eval_results)
                        run.finish()
                
                # Clean up
                del model
                del env
                del eval_env
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            results[policy_type] = policy_results
            
        return results

def main():
    # Initialize experiment manager
    experiment = ExperimentManager()
    
    # Run training and evaluation
    results = experiment.train_and_evaluate()
    
    # Save results
    results_path = os.path.join(experiment.base_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nTraining and evaluation complete!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()