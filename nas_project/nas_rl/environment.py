import numpy as np

class RLEnvironment:
    def __init__(self, max_layers=10):
        self.max_layers = max_layers
        self.current_step = 0
        self.architecture = []
        
        # Define Action Space
        # We will flatten the action space for simplicity in the Controller
        # Action = (Layer Type, Units, Activation)
        # But to make it easier for the RNN controller, we can have separate heads or a single large vocabulary.
        # Let's use a simplified vocabulary approach where each integer maps to a specific configuration.
        
        self.layer_types = ['dense', 'lstm', 'gru', 'rnn', 'dropout', 'batch_norm']
        self.units = [16, 32, 64, 128, 256]
        self.activations = ['relu', 'tanh', 'sigmoid', 'linear']
        
        # Construct vocabulary
        self.vocab = []
        for lt in self.layer_types:
            if lt in ['dense', 'lstm', 'gru', 'rnn']:
                for u in self.units:
                    for act in self.activations:
                        self.vocab.append({'type': lt, 'units': u, 'activation': act})
            elif lt == 'dropout':
                self.vocab.append({'type': lt, 'rate': 0.2}) # Fixed rate for simplicity
                self.vocab.append({'type': lt, 'rate': 0.5})
            elif lt == 'batch_norm':
                self.vocab.append({'type': lt})
        
        # Add STOP action
        self.vocab.append({'type': 'stop'})
        
        self.action_space_size = len(self.vocab)
        self.mode = 'full' # 'full' or 'dense_only'
        
    def set_mode(self, mode):
        self.mode = mode

    def set_max_layers(self, max_layers):
        self.max_layers = max_layers

    def reset(self):
        self.current_step = 0
        self.architecture = []
        return []

    def get_valid_actions(self):
        """
        Returns a list of valid action indices based on current state and mode.
        """
        valid_indices = []
        for i, action in enumerate(self.vocab):
            # Constraint 1: First layer must be Dense (User request)
            if self.current_step == 0:
                if action['type'] == 'dense':
                    valid_indices.append(i)
                continue # Skip other checks for step 0
            
            # Constraint 2: Mode 'dense_only'
            if self.mode == 'dense_only':
                if action['type'] in ['dense', 'dropout', 'batch_norm', 'stop']:
                    valid_indices.append(i)
                continue
            
            # Mode 'full' - allow everything
            valid_indices.append(i)
            
        return valid_indices

    def get_all_architectures(self, depth):
        """
        Generator that yields all valid architectures (lists of action indices) for a fixed depth.
        """
        import itertools
        
        # We need to generate sequences of length 'depth'.
        # But we must respect constraints:
        # 1. First layer must be Dense.
        # 2. Subsequent layers can be anything (in 'full' mode).
        
        # Get valid actions for step 0
        self.current_step = 0
        valid_start_indices = self.get_valid_actions()
        
        # Get valid actions for subsequent steps (assuming 'full' mode for exhaustive search)
        # We assume step > 0 allow all actions (except maybe STOP if we want fixed depth)
        # If we want fixed depth 'depth', we should NOT include 'stop' token in the middle.
        # So we filter out 'stop' token for the generation.
        
        non_stop_indices = [i for i, a in enumerate(self.vocab) if a['type'] != 'stop']
        
        # Create iterators
        # First layer: valid_start_indices
        # Rest layers: non_stop_indices
        
        iterators = [valid_start_indices] + [non_stop_indices] * (depth - 1)
        
        for p in itertools.product(*iterators):
            yield list(p)

    def step(self, action_index):
        """
        Args:
            action_index: Integer index into self.vocab
        Returns:
            next_state: The updated architecture (list of indices)
            reward: 0 (intermediate), calculated at end
            done: Boolean
            info: dict
        """
        if action_index < 0 or action_index >= self.action_space_size:
            raise ValueError(f"Invalid action index: {action_index}")

        layer_config = self.vocab[action_index]
        
        if layer_config['type'] == 'stop':
            done = True
        else:
            self.architecture.append(layer_config)
            self.current_step += 1
            done = self.current_step >= self.max_layers
        
        return self.architecture, 0, done, {}

    def get_architecture(self):
        return self.architecture

    def sample_action(self):
        return np.random.randint(0, self.action_space_size)
