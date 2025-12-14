import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class RLAgent:
    def __init__(self, action_space_size, embedding_dim=16, hidden_units=64, learning_rate=0.01):
        self.action_space_size = action_space_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        
        self.model = self._build_controller()
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        # Store memory for the current episode
        self.log_probs = []
        self.rewards = []

    def _build_controller(self):
        """
        Builds the Controller Network (RNN).
        Input: Sequence of action indices (integers).
        Output: Softmax probability distribution over the action space.
        """
        # Input is a sequence of integers (action indices)
        # We process them step-by-step or as a sequence.
        # Since we need to predict the NEXT action given previous actions,
        # we can feed the full history so far.
        
        inputs = layers.Input(shape=(None,), dtype=tf.int32)
        x = layers.Embedding(input_dim=self.action_space_size + 1, output_dim=self.embedding_dim)(inputs)
        # +1 for the start token or padding if needed.
        
        x = layers.LSTM(self.hidden_units)(x)
        outputs = layers.Dense(self.action_space_size, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def get_action(self, state, valid_actions=None):
        """
        Args:
            state: List of action indices taken so far.
            valid_actions: List of valid action indices. If None, all are valid.
        Returns:
            action_index: The chosen action.
        """
        # If state is empty, we can feed a special start token or just zeros.
        # Let's use a start token (action_space_size)
        if not state:
            input_seq = np.array([[self.action_space_size]])
        else:
            # Append start token at the beginning? Or just use the state.
            # Let's prepend start token.
            input_seq = np.array([[self.action_space_size] + state])
            
        # probs = self.model.predict(input_seq, verbose=0)[0]
        probs = self.model(input_seq, training=False)[0].numpy()
        
        # Apply Masking
        if valid_actions is not None:
            mask = np.zeros_like(probs)
            mask[valid_actions] = 1.0
            probs = probs * mask
            
            # Normalize
            if np.sum(probs) == 0:
                # Fallback if all probs are zero (shouldn't happen if model is initialized well, but possible)
                # Uniform over valid actions
                probs[valid_actions] = 1.0
            
            probs = probs / np.sum(probs)
        
        # Sample action
        action_index = np.random.choice(self.action_space_size, p=probs)
        
        return action_index

    def update_policy(self, states, actions, reward):
        """
        Updates the policy using the REINFORCE algorithm.
        Args:
            states: List of states (each state is a list of indices) for the episode.
            actions: List of action indices taken.
            reward: The final reward obtained.
        """
        # We assume one update per episode (Monte Carlo Policy Gradient)
        
        with tf.GradientTape() as tape:
            loss = 0
            for state, action in zip(states, actions):
                if not state:
                    input_seq = np.array([[self.action_space_size]])
                else:
                    input_seq = np.array([[self.action_space_size] + state])
                
                # Forward pass
                logits = self.model(input_seq, training=True) # (1, action_space)
                
                # Calculate log probability of the taken action
                # We want to MAXIMIZE reward, so we MINIMIZE -log_prob * reward
                
                # Cross entropy between actual action (one-hot) and predicted probs
                # is -log(prob_of_action).
                # So loss = CrossEntropy * Reward
                
                action_one_hot = tf.one_hot([action], self.action_space_size)
                ce_loss = tf.keras.losses.categorical_crossentropy(action_one_hot, logits)
                
                loss += ce_loss * reward
            
            # Average loss over the steps? Or sum?
            # Usually sum for the episode return.
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
