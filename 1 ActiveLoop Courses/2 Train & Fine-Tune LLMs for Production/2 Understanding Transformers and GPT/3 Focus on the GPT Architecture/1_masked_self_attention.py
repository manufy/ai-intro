import numpy as np

def self_attention(query, key, value, mask=None):
    # Compute attention scores
    scores = np.dot(query, key.T)
    
    if mask is not None:
        # Apply mask by setting masked positions to a large negative value
        scores = scores + mask * -1e9
    
    # Apply softmax to obtain attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Compute weighted sum of value vectors
    output = np.dot(attention_weights, value)
    
    return output