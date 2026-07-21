import numpy as np

data = np.load('expert_demos.npz')

print(data.files)
print(data['actions'])