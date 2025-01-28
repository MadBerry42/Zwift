import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\Processed Data\\Input to model"
file1 = pd.read_excel(f'{path}\\000_input_file.xlsx')
file2 = pd.read_excel(f'{path}\\003_input_file.xlsx')
file3 = pd.read_excel(f'{path}\\004_input_file.xlsx')
file4 = pd.read_excel(f'{path}\\006_input_file.xlsx')
file5 = pd.read_excel(f'{path}\\007_input_file.xlsx')
file6 = pd.read_excel(f'{path}\\008_input_file.xlsx')
file7 = pd.read_excel(f'{path}\\009_input_file.xlsx')
file8 = pd.read_excel(f'{path}\\010_input_file.xlsx')
file9 = pd.read_excel(f'{path}\\011_input_file.xlsx')
file10 = pd.read_excel(f'{path}\\012_input_file.xlsx')
file11 = pd.read_excel(f'{path}\\013_input_file.xlsx')
file12 = pd.read_excel(f'{path}\\014_input_file.xlsx')
file13 = pd.read_excel(f'{path}\\015_input_file.xlsx')
file14 = pd.read_excel(f'{path}\\016_input_file.xlsx')
data_or = pd.concat([file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11, file12, file13, file14], ignore_index=True)

# Normalize features to [0, 1] range
features = ['Heart Rate', 'RPE', 'Power hc', 'Power bc']#, 'hr_mean', 'hr_min', 'hr_sd', 'hr_energy', 'hr_entropy', 'hr_iqr', 'hr_kurtosis', 'hr_skewness', 'hr_mad']
data = data_or.copy()
data[features] = (data[features] - data[features].min()) / (data[features].max() - data[features].min())
X = data[features].values
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


# Define Generator with 3 fully connected layers
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),  # LeakyReLU instead of ReLU to enhance gradient flow
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Tanh to ensure output is in the range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Define Discriminator with 3 fully connected layers and dropout for regularization
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Dropout layer for regularization
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Sigmoid to output probability of being real
        )

    def forward(self, x):
        return self.model(x)
    

# Initialize generator and discriminator
input_dim = 32  # Dimension of the noise vector
output_dim = X_train_tensor.shape[1]  # Number of features (same as input feature size)
G = Generator(input_dim, output_dim)
D = Discriminator(output_dim)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0001)  # Lower learning rate for D

# Prepare subset of data
subset_size = 750
subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
X_train_subset = X_train[subset_indices]

# Convert subset to PyTorch tensor
X_train_subset_tensor = torch.tensor(X_train_subset, dtype=torch.float32)


# Training loop for Vanilla GAN
def train_vanilla_gan(epochs, batch_size):
    data_loader = DataLoader(TensorDataset(X_train_subset_tensor), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        g_error = 0.0
        d_error = 0.0
        for i, (real_data,) in enumerate(data_loader):
            # Dynamically set batch size
            batch_size = real_data.size(0)
            noise = torch.randn(batch_size, input_dim)
            fake_data = G(noise)

            # Add small Gaussian noise to real data
            real_data_noisy = real_data + 0.05 * torch.randn_like(real_data)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.full((batch_size, 1), 0.9)  # Label smoothing for real data
            fake_labels = torch.zeros(batch_size, 1)

            # Compute Discriminator loss
            loss_D_real = criterion(D(real_data_noisy), real_labels)
            loss_D_fake = criterion(D(fake_data.detach()), fake_labels)
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            d_error += loss_D.item()

            # Train Generator
            optimizer_G.zero_grad()
            loss_G = criterion(D(fake_data), real_labels)  # Generator aims to make fake data look real
            loss_G.backward()
            optimizer_G.step()
            g_error += loss_G.item()

        # Average errors for epoch
        g_error /= len(data_loader)
        d_error /= len(data_loader)

        # Print epoch losses
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss D: {d_error:.8f}, Loss G: {g_error:.8f}")


# Train Vanilla GAN with specified parameters
train_vanilla_gan(epochs=2000, batch_size=32)


# Generate synthetic data
latent_points = torch.randn(750, input_dim)  # 750 synthetic samples
synthetic_data = G(latent_points)

# Convert to numpy array for easier handling and save the synthetic data
synthetic_data = synthetic_data.detach().numpy()


# Denormalize synthetic data to match the original range
for i in range(synthetic_data.shape[1]):
    synthetic_data[:, i] = synthetic_data[:, i] * (data_or[features].iloc[:, i].max() - data_or[features].iloc[:, i].min()) + data_or[features].iloc[:, i].min()

synthetic_data_df = pd.DataFrame(synthetic_data, columns=features)


# Save synthetic data
synthetic_data_df.to_csv(f'{path}\\Fake_data3.csv', index=False)
print("Synthetic data saved as synthetic_data.csv")