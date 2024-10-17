import matplotlib.pyplot as plt

# 示例数据
epochs = [1, 2, 3, 4, 5]
mean_dsc_scores = [0.6, 0.65, 0.7, 0.68, 0.7]

plt.figure(figsize=(10, 5))
plt.plot(epochs, mean_dsc_scores, marker='x', linestyle='-', color='r')
plt.xlabel('Epochs')
plt.ylabel('Mean DSC Scores')
plt.title('Mean DSC Scores over Epochs')
plt.grid(True)
plt.show()

import seaborn as sns
import pandas as pd

# 示例数据
data = {
    'Epoch': ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'],
    'Mean DSC Scores': [0.6, 0.65, 0.7, 0.68, 0.7]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 5))
sns.boxplot(x='Epoch', y='Mean DSC Scores', data=df)
plt.xlabel('Epochs')
plt.ylabel('Mean DSC Scores')
plt.title('Box Plot of Mean DSC Scores')
plt.grid(True)
plt.show()
