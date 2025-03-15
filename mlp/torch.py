from torch.nn import Linear, Module
import torch.nn.functional as F
import torch

class MLP(Module):
    def __init__(self, infeats, fc_feats, units):
        super(MLP, self).__init__()
        self.inlayer = Linear(infeats, fc_feats)
        self.fc = Linear(fc_feats, fc_feats)
        self.outlayer = Linear(fc_feats, units)
        
    def forward(self, x):
        x = torch.relu(self.inlayer(x))
        x = torch.relu(self.fc(x))
        x = self.outlayer(x)
        return x
    
model = MLP(768, 1024, 3)
data = torch.rand(100, 768)
one_hot_labels = F.one_hot(
    torch.tensor([i % 3 for i in range(100)]), num_classes=3,
).type(torch.float32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for i in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, one_hot_labels)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Epoch {i}: {loss.item()}")
        
accuracy = (torch.argmax(out, dim=1) == torch.argmax(one_hot_labels, dim=1)).sum().item() / 100
print(f"Accuracy: {accuracy}")
