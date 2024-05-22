import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class MerwinDataset(Dataset):
    def __init__(self, size, image_dim, text_dim):
        self.size = size
        self.image_dim = image_dim
        self.text_dim = text_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(self.image_dim)
        text = torch.randn(self.text_dim)
        return {'image': image, 'text': text}

class MerwinEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MerwinEncoder, self).__init__()
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        return self.fc(x)

class SimpleCrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimpleCrossAttention, self).__init__()
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y):
        q = self.proj_q(x)
        k = self.proj_k(y)
        v = self.proj_v(y)
        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5), dim=-1)
        return torch.matmul(attn_weights, v)


# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, global_logit_scale, local_logit_scale, batch_size, temperature, local_cross_attention,
                 predictor):
        super(CombinedLoss, self).__init__()
        self.global_loss = GlobalAlignmentLoss(global_logit_scale, batch_size)
        self.local_loss = LocalAlignmentLoss(local_cross_attention, predictor, local_logit_scale)
        self.nt_xent_loss = NT_XentLoss(batch_size, temperature)

    def forward(self, image_embed, text_embed, local_image_embed_stacks, local_text_embed_stacks, z_i, z_j):
        global_loss = self.global_loss(image_embed, text_embed)
        local_image_loss, local_text_loss = self.local_loss(local_image_embed_stacks, local_text_embed_stacks)
        nt_xent_loss = self.nt_xent_loss(z_i, z_j)
        total_loss = global_loss + local_image_loss + local_text_loss + nt_xent_loss
        return total_loss


# Simulated data
batch_size = 16
embed_dim = 128
temperature = 0.5
dataset = MerwinDataset(1000, 256, 256)
dataloader = MerwinLoader(dataset, batch_size=batch_size, shuffle=True)

# Simulated encoders and attention module
image_encoder = MerwinEncoder(embed_dim)
text_encoder = MerwinEncoder(embed_dim)
local_cross_attention = SimpleCrossAttention(embed_dim)
predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                          nn.ReLU(inplace=True),
                          nn.Linear(embed_dim // 2, embed_dim))

# Combined loss function
loss_fn = CombinedLoss(global_logit_scale=0.07, local_logit_scale=0.07, batch_size=batch_size, temperature=temperature,
                       local_cross_attention=local_cross_attention, predictor=predictor)

optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(
    local_cross_attention.parameters()) + list(predictor.parameters()), lr=1e-4)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        image = batch['image']
        text = batch['text']

        image_embed = image_encoder(image)
        text_embed = text_encoder(text)

        local_image_embed = image_embed.view(batch_size, 1, -1)  # Simulated local features
        local_text_embed = text_embed.view(batch_size, 1, -1)  # Simulated local features

        z_i = torch.randn(batch_size, embed_dim)  # Simulated embeddings for NT_XentLoss
        z_j = torch.randn(batch_size, embed_dim)  # Simulated embeddings for NT_XentLoss

        loss = loss_fn(image_embed, text_embed, local_image_embed, local_text_embed, z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")


