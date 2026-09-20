import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#
#         self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim))
#         self.b1 = nn.Parameter(torch.randn(hidden_dim))
#
#         self.w2 = nn.Parameter(torch.randn(output_dim, hidden_dim))
#         self.b2 = nn.Parameter(torch.randn(output_dim))
#
#     def forward(self, x):
#         # Linear 1
#         x = torch.matmul(x, self.w1.T)
#         x = x + self.b1
#
#         # ReLU
#         x = torch.relu(x)
#
#         # Linear 2
#         x = torch.matmul(x, self.w2.T)
#         x = x + self.b2
#
#         return x

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 使用融合后的 LinearReLU 算子模块
        self.fc1_relu = nni.LinearReLU(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1_relu(x)
        x = self.fc2(x)
        return x

# 示例
model = MLP(
    input_dim=10,
    hidden_dim=64,
    output_dim=3
)

x = torch.randn(32, 10)  # batch_size=32
y = model(x)

print(model)

print(y.shape)  # torch.Size([32, 3])

# 构造一个示例输入
dummy_input = torch.randn(1, 10)

# 导出 ONNX
torch.onnx.export(
    model,
    (dummy_input,),
    "d0918-mlp-v3.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=True,
    optimize=True,
)

print("ONNX 模型已导出：mlp.onnx")
