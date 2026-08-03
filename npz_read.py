import numpy as np

data = np.load('expert_demos.npz')

print(data.files)
print(data['episode_index'][:30])
print(data['episode_starts'][:30])
print(len(data['observation']))