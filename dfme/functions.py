import torch
import tensorflow as tf
import numpy as np

def tf_to_torch(tf_tensor):
    torch_tensor = torch.tensor(tf_tensor.numpy())
    # torch_tensor = torch_tensor.to(device)
    return torch_tensor

def torch_to_tf(torch_tensor):
    tf_tensor = torch_tensor.detach().cpu().numpy()
    tf_tensor = tf_tensor.reshape(tf_tensor.shape[0], tf_tensor.shape[2], tf_tensor.shape[3], tf_tensor.shape[1])
    tf_tensor = np.expand_dims(tf_tensor, axis=0)
    tf_tensor = tf.convert_to_tensor(tf_tensor, dtype=tf.float32)
    return tf_tensor
