🏷 #训练技术 #强化学习 #优化 #进阶

# PPO（Proximal Policy Optimization，近端策略优化）

> 关联：[[K2-技术方法与实现/训练技术/RLHF人类反馈强化学习|RLHF人类反馈强化学习]]、[[K1-基础理论与概念/核心概念/奖励模型（Reward Model）|奖励模型]]、[[SFT（Supervised Fine-Tuning，监督微调）|SFT]]

---

## 概念定义
PPO是一种稳定、实现简单的强化学习算法，常用于RLHF中的策略优化阶段。其核心思想是通过“截断的目标函数（clip）”或“KL约束/惩罚”限制每次更新的步幅，避免策略发生剧烈偏移。

---

## 在RLHF中的作用
- 输入：由[[奖励模型（Reward Model）|奖励模型]]对模型输出打分得到的回报信号。
- 目标：在保持与SFT参考分布相近的前提下，提高被人类偏好的响应概率。
- 结果：更新后的策略模型更符合人类意图与安全边界。

---

## 关键机制
- Clip目标：限制概率比率 r_t 的更新幅度（如 1±0.1/0.2）。
- KL约束/惩罚：控制新策略与参考策略间的KL散度，保障稳定性与可控性。
- 优势估计：常配合GAE使用，提高低方差、稳定训练。

---

## 典型训练流程（RLHF场景）
1) 采样：策略模型对prompt生成多个响应；
2) 打分：[[奖励模型（Reward Model）|奖励模型]]为每个响应给出奖励；
3) 估计：计算优势（如GAE），并加入KL正则或clip；
4) 优化：按小步多轮（epochs）更新策略参数；
5) 迭代：重复采样-优化直至收敛与指标达标。

---

## 常见超参数参考
- 学习率：1e-6 ~ 1e-5（通常小于SFT阶段）
- Clip范围：0.1 ~ 0.2
- 每步样本数 / 批次：依资源与上下文长度调整
- 更新轮次（epochs）：1 ~ 4
- KL目标/惩罚系数：基于经验与稳定性需求调节

---

## 工程实践建议
- 使用参考模型（frozen SFT）计算KL约束，防止策略漂移；
- 归一化优势、梯度裁剪，提升数值稳定性；
- 监控奖励趋势与人评一致性，避免“奖励黑客”。

---

## TRL 示例配置（简化）
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name="your-sft-model",
    learning_rate=1e-6,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    init_kl_coef=0.1,
    target_kl=0.1,
    cliprange=0.2,
    seed=42,
)

policy = AutoModelForCausalLM.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)  # 冻结参考
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
ppo_trainer = PPOTrainer(config, policy, ref_model, tokenizer)

# 训练循环（示意）
for batch in prompt_dataloader:
    queries = batch["prompt_text"]
    # 1) 生成
    responses = ppo_trainer.generate(queries, max_new_tokens=256)
    # 2) 奖励打分（外部Reward Model）
    rewards = reward_model_score(queries, responses)  # 张量/列表
    # 3) PPO更新（自动含KL约束/clip）
    stats = ppo_trainer.step(queries, responses, rewards)
```

---

## 超参数小抄（经验值）
- 学习率：小模型可取 1e-6~2e-6；大模型倾向更小
- KL控制：`target_kl≈0.05~0.2`；开局偏大、后期逐步减小
- clip：`0.1~0.2`；过大引入不稳定，过小收敛慢
- 批大小：受上下文长度与显存影响，建议按token预算反推

---

## 与相关方法的关系
- 与[[RLHF人类反馈强化学习|RLHF]]：RLHF第三阶段的默认优化器。
- 与[[奖励模型（Reward Model）|奖励模型]]：PPO依赖奖励模型提供的回报信号进行更新。
- 与[[DPO直接偏好优化|DPO]]：DPO跳过PPO与奖励模型，直接从偏好数据学习。

---

## 一句话总结
PPO用小步稳健的策略更新，把“奖励模型给分”转化为“更符合人类偏好的策略”。
