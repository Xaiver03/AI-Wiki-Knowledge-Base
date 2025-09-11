🏷 #核心概念 #对齐 #偏好学习 #进阶

# 奖励模型（Reward Model）

> 关联：[[K2-技术方法与实现/训练技术/RLHF人类反馈强化学习|RLHF人类反馈强化学习]]、[[K2-技术方法与实现/训练技术/PPO（Proximal Policy Optimization，近端策略优化）|PPO]]、[[SFT（Supervised Fine-Tuning，监督微调）|SFT]]、[[K2-技术方法与实现/训练技术/DPO直接偏好优化|DPO]]

---

## 概念定义
奖励模型用于对模型输出的质量进行评分，学习“人类更喜欢哪种回答”的偏好函数。在RLHF中，它把人类偏好数据转化为可用于优化策略模型的标量奖励信号。

---

## 在RLHF中的作用
- 将人类偏好（排序/胜负对）学习为打分函数 R(prompt, response) → 实数分数。
- 在[[RLHF人类反馈强化学习|RLHF]]的第三阶段由[[PPO（Proximal Policy Optimization，近端策略优化）|PPO]]读取该分数，推动策略模型朝“更被人类偏好”的方向优化。
- 可扩展为过程奖励模型（PRM），对思维链/步骤级质量打分；或结果奖励模型（ORM），对最终答案打分。

---

## 数据与训练方式
- 数据形式：偏好对或排序（prompt, response_chosen, response_rejected）。
- 训练目标：最大化被选回答的分数、最小化被拒回答的分数（常用成对比较/Bradley–Terry 风格目标）。
- 架构选择：通常沿用[[Transformer架构原理|Transformer]]骨干并在顶层接回归头或比较损失。
- 评价与监控：与人工评分/红队集一致性，防止“奖励黑客”（Reward Hacking）。

---

## 损失函数与训练目标（细化）
- 成对Logistic损失（常用）：
  - 记 rθ(x, y) 为奖励模型对回答 y 的打分；给定偏好对 (y⁺, y⁻)，优化
  - L(θ) = − E[ log σ(rθ(x, y⁺) − rθ(x, y⁻)) ]，σ 为 Sigmoid
  - 直观：被选回答比分数差越大，损失越小
- 排序/多项式扩展：可将同一prompt的多回答做List-wise目标（如ListNet、Softmax温度加权）
- Hinge/Margin变体：L = max(0, m − (r(y⁺) − r(y⁻)))，更“硬”的间隔约束
- 正则与校准：L2正则、温度/偏置校准，避免分数外推过度

---

## 实现伪码（Pairwise）
```
for batch in dataloader:  # (x, y_pos, y_neg)
    s_pos = reward_model.score(x, y_pos)  # rθ(x, y⁺)
    s_neg = reward_model.score(x, y_neg)  # rθ(x, y⁻)
    delta = s_pos - s_neg
    loss = -logsigmoid(delta).mean()
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
```

### PyTorch示意
```python
import torch, torch.nn as nn

class RewardHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # Transformer encoder/LM
        self.scorer = nn.Linear(backbone.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        s = self.scorer(h[:, -1])  # 取末token或做池化
        return s.squeeze(-1)

def pairwise_loss(s_pos, s_neg):
    return -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()
```

---

## 设计与实践要点
- 数据质量：覆盖多场景、避免标注偏差；定期去噪与再采样。
- 稳定性：监控分布偏移；对极端/越界输出加约束或规则过滤。
- 结合KL约束：在策略优化时配合KL正则，避免策略远离SFT分布导致奖励模型外推失真。

---

## 与相关方法的关系
- 与[[RLHF人类反馈强化学习|RLHF]]：RLHF必备组件之一，提供优化信号。
- 与[[PPO（Proximal Policy Optimization，近端策略优化）|PPO]]：PPO使用奖励模型的分数作为回报进行策略更新。
- 与[[DPO直接偏好优化|DPO]]：DPO直接从偏好数据学习，省去独立奖励模型与PPO阶段。

---

## 开源与资料
- Hugging Face TRL：Reward Model 与 PPO 训练范式示例
- Open-source RLHF pipelines：包含偏好数据构建与奖励模型训练脚本

---

## 评估与诊断
- 一致性：与人评一致率、Kendall τ、Spearman ρ
- 鲁棒性：越界/对抗样本的评分稳定性
- 反作弊：监控高分低质样本占比，发现奖励黑客迹象

---

## 一句话总结
奖励模型把“人类偏好”转成“可优化的分数”。它是RLHF的桥梁，使策略优化能够贴合人类价值与偏好。
