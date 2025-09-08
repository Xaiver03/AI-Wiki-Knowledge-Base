
🏷 #入门 #训练 #NLP

---

### **✅ 什么是 Few-shot？**

  

**Few-shot learning** 是一种**只提供很少几个示例（shots）就能完成任务的学习方式**。现代基于[[Transformer架构原理|Transformer架构]]的大语言模型展现出了强大的Few-shot学习能力。

这些“示例”通常是在 Prompt（提示）中直接给出的输入-输出对，模型基于这些少量示例推理并生成答案。

  

例如：

```
例子1：
Q: 巴黎是哪个国家的首都？
A: 法国

例子2：
Q: 东京是哪个国家的首都？
A: 日本

问题：
Q: 罗马是哪个国家的首都？
A: ??
```

这就是一个标准的 Few-shot Prompt，模型需要根据前两个例子推理出答案。

---

### **🧠 与 Few-shot 对应的其他几种 Shot：**

|**名称**|**示例数量**|**说明**|
|---|---|---|
|**Zero-shot**|0|不提供任何示例，模型仅根据任务指令生成答案。👉 对模型泛化能力要求高。|
|**One-shot**|1|只提供一个示例作为参考。👉 可略微提示任务格式或逻辑。|
|**Few-shot**|2–5 左右|提供少量示例，模型据此推理。👉 是提示学习（prompt learning）的经典形式。|
|**Many-shot**（非正式）|>10|提供较多示例，容易达到 Token 上限。👉 在计算上更昂贵。|

> 注：这里的“shot”并不指模型训练过程中的样本，而是**推理阶段输入的 Prompt 中展示的示例数量**。

---

### **🧪 Few-shot 与传统学习方法的比较：**

|**模型类型**|**学习方式**|**示例来源**|**应用场景**|
|---|---|---|---|
|传统监督学习|大量训练样本|数据集|分类、回归等任务|
|Fine-tune 微调模型|预训练 + 少量训练数据微调|开发者准备|定制模型|
|Prompt-based Few-shot|不训练，直接用示例提示|用户提供示例|即用型任务，灵活高效|

---

### **🌟 延伸概念：**

- **Prompt Engineering**（提示工程）：如何设计优质的 few-shot prompt 是一门技巧活。
    
- **In-context Learning**（上下文学习）：few-shot learning 本质上是让模型**在上下文中学习规则与模式**，而不是重新训练模型参数。
    

---

### **✅ 总结**

```
graph TD
    A[Zero-shot<br>0示例] --> B[One-shot<br>1示例]
    B --> C[Few-shot<br>2-5个示例]
    C --> D[Many-shot<br>10+示例]
    style A fill:#FFEEE0
    style B fill:#FFF2CC
    style C fill:#DFF2BF
    style D fill:#CFE2F3
```

- **Zero-shot：** 没有例子，靠理解能力；
    
- **Few-shot：** 给几个例子，靠类比与模式识别；
    
- **Fine-tune：** 改模型参数，记住任务方式；
    
- **Few-shot更像“在用”的智能，而Fine-tune更像“学会”的能力。**
    

---

如需我再展开提示工程技巧、Few-shot与Chain-of-Thought等思维链结合方式，欢迎继续提问！