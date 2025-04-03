import torch
import torch.nn as nn
import torch.optim as optim

# 定义多标签分类器
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        参数：
        - input_dim: 输入特征的维度
        - hidden_dim: 隐藏层单元数
        - output_dim: 输出标签数量，每个标签对应一个神经元
        """
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 输出层，返回 logits

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)  # 输出 logits，不直接经过 Sigmoid
        return logits

# 示例使用
if __name__ == "__main__":
    # 超参数设置
    batch_size = 8
    input_dim = 10      # 输入特征数量
    hidden_dim = 20     # 隐藏层单元数
    output_dim = 5      # 标签数，例如有5个标签，每个样本可同时属于多个类别

    # 构造随机输入数据和目标标签
    inputs = torch.randn(batch_size, input_dim)
    # 目标标签为二值，多标签问题中每个样本对应一个 0/1 的向量
    targets = torch.randint(0, 5, (batch_size, output_dim)).float()

    # 实例化模型
    model = MultiLabelClassifier(input_dim, hidden_dim, output_dim)

    # 使用 BCEWithLogitsLoss 作为损失函数，此函数内部包含 Sigmoid 操作
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 简单训练循环
    model.train()
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(inputs)  # 模型输出 logits
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # 模型评估，获取每个标签的预测概率
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        # 根据设定阈值（例如 0.5）将概率转换为二值标签
        predicted_labels = (probabilities > 0.5).float()
        print("预测概率：")
        print(probabilities)
        print("预测标签：")
        print(predicted_labels)
