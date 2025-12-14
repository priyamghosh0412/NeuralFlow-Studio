import tensorflow as tf
from tensorflow.keras import layers, models

class ModelBuilder:
    def __init__(self, input_shape, output_shape, problem_type):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.problem_type = problem_type  # 'classification' or 'regression'

    def build_model(self, architecture):
        """
        Builds a Keras model from a list of layer specifications.
        """
        model = models.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(input_shape=self.input_shape))
        
        # Track current tensor rank (2 for (batch, features), 3 for (batch, time, features))
        # We assume input_shape is (features,) -> Rank 2 (batch, features)
        # If input_shape is (time, features) -> Rank 3.
        
        current_rank = len(self.input_shape) + 1
        current_shape = self.input_shape # Tuple of dimensions excluding batch

        for i, layer_spec in enumerate(architecture):
            layer_type = layer_spec.get('type')
            
            try:
                # Handle RNN layers (LSTM, GRU, SimpleRNN)
                if layer_type in ['lstm', 'gru', 'rnn']:
                    # RNNs require 3D input (batch, time, features)
                    if current_rank == 2:
                        # Reshape 2D -> 3D: (batch, features) -> (batch, 1, features)
                        # We can use Reshape layer.
                        # We need to know the feature dimension.
                        # Keras Reshape target_shape does not include batch.
                        # If current_shape is (features,), target is (1, features).
                        target_shape = (1, -1) # -1 infers the dimension
                        model.add(layers.Reshape(target_shape))
                        current_rank = 3
                    
                    # Determine return_sequences
                    # If next layer is RNN, we need sequence output (3D).
                    # If next layer is Dense, we might want 2D (False) or 3D (True).
                    # For simplicity: if next is RNN, return_sequences=True.
                    # If next is Dense/Output, return_sequences=False (flatten time).
                    
                    next_is_rnn = False
                    if i < len(architecture) - 1:
                        next_type = architecture[i+1].get('type')
                        if next_type in ['lstm', 'gru', 'rnn']:
                            next_is_rnn = True
                    
                    # If it's the last layer, we usually want to flatten for the head, 
                    # unless the head handles 3D (but our head assumes 2D usually).
                    # So if last layer, return_sequences=False.
                    if i == len(architecture) - 1:
                        return_sequences = False
                    else:
                        return_sequences = next_is_rnn

                    units = layer_spec.get('units', 32)
                    activation = layer_spec.get('activation', 'tanh')
                    
                    if layer_type == 'lstm':
                        model.add(layers.LSTM(units=units, return_sequences=return_sequences, activation=activation))
                    elif layer_type == 'gru':
                        model.add(layers.GRU(units=units, return_sequences=return_sequences, activation=activation))
                    elif layer_type == 'rnn':
                        model.add(layers.SimpleRNN(units=units, return_sequences=return_sequences, activation=activation))
                    
                    if return_sequences:
                        current_rank = 3
                    else:
                        current_rank = 2

                elif layer_type == 'dense':
                    # Dense works on 2D and 3D.
                    # If 3D (batch, time, feat) -> (batch, time, units)
                    # If 2D (batch, feat) -> (batch, units)
                    model.add(layers.Dense(
                        units=layer_spec.get('units', 32),
                        activation=layer_spec.get('activation', 'relu')
                    ))
                    # Rank doesn't change

                elif layer_type == 'dropout':
                    model.add(layers.Dropout(rate=layer_spec.get('rate', 0.2)))

                elif layer_type == 'batch_norm':
                    model.add(layers.BatchNormalization())

            except Exception as e:
                print(f"Error adding layer {layer_spec}: {e}")
                return None

        # Output Head
        # Ensure we are 2D before the head
        if current_rank == 3:
            model.add(layers.Flatten())
            
        if self.problem_type == 'classification':
            num_classes = self.output_shape[0]
            if num_classes == 1:
                 model.add(layers.Dense(1, activation='sigmoid'))
            else:
                 model.add(layers.Dense(num_classes, activation='softmax'))
            
            loss = 'sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
            metrics = ['accuracy']

        elif self.problem_type == 'regression':
            output_dim = self.output_shape[0]
            model.add(layers.Dense(output_dim, activation='linear'))
            loss = 'mse'
            metrics = ['mae', 'mse']

        model.compile(optimizer='adam', loss=loss, metrics=metrics)
        return model
