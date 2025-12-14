import numpy as np
from .builder import ModelBuilder
from .environment import RLEnvironment
from .controller import RLAgent
from .trainer import Trainer

class NASFramework:
    def __init__(self, input_shape, output_shape, problem_type, max_layers=5, max_epochs=5):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.problem_type = problem_type
        
        self.env = RLEnvironment(max_layers=max_layers)
        self.builder = ModelBuilder(input_shape, output_shape, problem_type)
        self.trainer = Trainer(self.builder, epochs=max_epochs)
        self.agent = RLAgent(self.env.action_space_size)
        
        self.best_architecture = None
        self.best_reward = -float('inf')
        self.best_metrics = {}
        self.best_model = None  # Store the actual trained model
        self.history = []

    def search(self, X_train, y_train, X_val, y_val, target_metric=0.95, patience=5, min_episodes=5, strategy='rl', callback=None, stop_signal=None):
        """
        Runs the NAS search loop using Iterative Depth Search with Dynamic Expansion.
        
        Args:
            target_metric: Threshold to stop search (Accuracy for classification, 1-MAPE for regression).
            patience: Number of episodes to wait without improvement before increasing depth.
            min_episodes: Minimum episodes to run at each depth before considering moving on.
            strategy: 'rl' (Reinforcement Learning) or 'exhaustive' (Systematic Search).
            callback: Optional function to call after each episode with status dict.
            stop_signal: Optional callable that returns True if search should stop.
        """
        print(f"Starting NAS Search (Target: {target_metric}, Strategy: {strategy}, Patience: {patience})...")
        
        limit_max_layers = self.env.max_layers # The absolute limit
        
        # Start from depth 2
        for depth in range(2, limit_max_layers + 1):
            print(f"\n==========================================")
            print(f" Exploring Depth: {depth}")
            print(f"==========================================")
            
            self.env.set_max_layers(depth)
            self.env.set_mode('full') 
            
            # Depth-specific variables
            best_reward_this_depth = -float('inf')
            patience_counter = 0
            episodes_run_this_depth = 0
            
            if strategy == 'exhaustive':
                # Exhaustive: We yield ALL architectures. 
                # Patience doesn't apply the same way, but we can stop if we hit target.
                iterator = self.env.get_all_architectures(depth)
            else:
                # RL: We run indefinitely until patience runs out
                # We use a large range as a safety cap, effectively infinite
                iterator = range(1000) 

            for i, item in enumerate(iterator):
                # Check Stop Signal
                if stop_signal and stop_signal():
                    print("\n!!! Search Stopped by User !!!")
                    return

                # Dynamic Stopping Condition (RL Only)
                if strategy == 'rl':
                    if episodes_run_this_depth >= min_episodes:
                        if patience_counter >= patience:
                            print(f"\nPerformance plateaued at Depth {depth} (Patience {patience} reached).")
                            print(">> Deciding to expand to next depth...")
                            break
                
                episode = i # Local index
                
                state = self.env.reset()
                done = False
                
                episode_states = []
                episode_actions = []
                
                # 1. Generate Architecture
                if strategy == 'exhaustive':
                    # item is the list of action indices
                    action_indices = item
                    for action_index in action_indices:
                        episode_states.append(list(state))
                        episode_actions.append(action_index)
                        _, _, done, _ = self.env.step(action_index)
                        state.append(action_index)
                else:
                    # RL Strategy
                    step_counter = 0
                    while not done:
                        valid_actions = self.env.get_valid_actions()
                        print(f"DEBUG: Step {step_counter}, Valid Actions: {len(valid_actions)}")
                        
                        action_index = self.agent.get_action(state, valid_actions)
                        print(f"DEBUG: Action Chosen: {action_index}")
                        
                        episode_states.append(list(state))
                        episode_actions.append(action_index)
                        
                        _, _, done, _ = self.env.step(action_index)
                        print(f"DEBUG: Step Done: {done}")
                        
                        state.append(action_index)
                        step_counter += 1
                        
                        if step_counter > 100:
                            print("DEBUG: Emergency Break - Infinite Loop Detected")
                            break
                
                # 2. Build and Train
                architecture_dicts = self.env.get_architecture()
                print(f"Ep {episodes_run_this_depth+1} | Depth {len(architecture_dicts)} | Arch: {[l['type'] for l in architecture_dicts]}")
                
                if callback:
                    callback({
                        'episode': episodes_run_this_depth,
                        'depth': depth,
                        'status': 'training',
                        'architecture': [l['type'] for l in architecture_dicts],
                        'best_reward': self.best_reward
                    })

                reward, metrics, trained_model = self.trainer.train_and_evaluate(
                    architecture_dicts, X_train, y_train, X_val, y_val
                )
                
                print(f"  Reward: {reward:.4f}, Metrics: {metrics}")
                
                # 3. Update Agent
                self.agent.update_policy(episode_states, episode_actions, reward)
                
                # 4. Track Best & Patience
                episodes_run_this_depth += 1
                
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_architecture = architecture_dicts
                    self.best_metrics = metrics
                    self.best_model = trained_model  # Save the trained model
                    print(f"  ** New Global Best Model! Reward: {self.best_reward:.4f} **")
                
                # Update Depth Best
                if reward > best_reward_this_depth:
                    best_reward_this_depth = reward
                    patience_counter = 0 # Reset patience
                    print(f"  (Depth Best Improved)")
                else:
                    patience_counter += 1
                    # print(f"  (No Improvement. Patience: {patience_counter}/{patience})")
                
                # 5. Callback
                if callback:
                    status = {
                        'episode': episodes_run_this_depth,
                        'depth': depth,
                        'architecture': architecture_dicts,  # Send full dicts with units/activation
                        'architecture_summary': [l['type'] for l in architecture_dicts],  # Keep simple view too
                        'reward': reward,
                        'metrics': metrics,
                        'best_reward': self.best_reward,
                        'best_architecture': self.best_architecture if self.best_architecture else []
                    }
                    callback(status)

                self.history.append({
                    'depth': depth,
                    'reward': reward,
                    'metrics': metrics,
                    'architecture': architecture_dicts
                })
                
                # Check Global Threshold
                if self.best_reward >= target_metric:
                    print(f"\n!!! Target Metric Threshold Reached ({self.best_reward:.4f} >= {target_metric}) !!!")
                    print("Stopping Search.")
                    print(f"Best Architecture: {self.best_architecture}")
                    return

            # End of Depth Loop
            print(f"Finished exploring Depth {depth}. Best Reward this depth: {best_reward_this_depth:.4f}")
            
        print("\nMax Layers Limit Reached.")
        print(f"Best Reward: {self.best_reward}")
        print(f"Best Architecture: {self.best_architecture}")

    def _run_episodes(self, n_episodes, X_train, y_train, X_val, y_val, start_episode):
        # Deprecated
        pass

    def get_best_model(self):
        """Returns the trained best model found during search."""
        return self.best_model
