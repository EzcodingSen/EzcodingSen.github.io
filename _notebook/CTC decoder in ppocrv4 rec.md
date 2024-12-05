---
title: 'CTC decoder in ppocrv4 rec'
date: 2024-12-04
permalink: /notebook/2024/12/CTC decoder in ppocrv4 rec/
tags:
  - OCR
---

ppocrv4 文本识别模型中采用的CTC解码器解析。

CTC 解码器的理论：
======

CTC（Connectionist Temporal Classification）解码器是一种专为处理输入长度和输出序列长度不一致的序列建模问题而设计的算法。它特别适用于语音识别、文本识别（如OCR）等领域。他的基本思想是从一个输入序列 
𝑋中预测一个对齐的输出序列𝑌，同时**避免显式地对每个输入和输出时间步进行对齐**。  
怎么理解这个显式的对齐呢？举个例子：输入是一段语音，输出是文字。显示的对齐则需要手动标记每个语音帧的输出字符。显然这是非常繁琐昂贵的。为此，CTC引入了一个特殊的 "blank" 标签（通常记为 "−" 或 "𝜖"），用于表示**无输出或忽略**的时间步，从而允许输出序列的长度小于输入序列。 

CTC的核心步骤
------

### 1.对齐路径（Alignment Paths）  
CTC定义了所有可能的对齐路径，用来从输入序列生成目标输出序列。这些路径包括目标序列插入 "blank" 标签后的所有可能时间步扩展。例如：  
输入序列长度：5  
输出序列：𝐴𝐵  
可能的路径是：A - - B / A - B - / - A B -等。  
### 2.路径概率计算  
给定模型输出的每个时间步上的概率分布𝑃(𝑐<sub>𝑡</sub>∣𝑡)（即预测某字符𝑐<sub>𝑡</sub>在时间𝑡的概率），CTC将每个𝑃(𝑐<sub>𝑡</sub>∣𝑡)乘起来，得到每条路径的概率:  

$$
P(\pi \mid X) = \prod_{t=1}^{T} P(c_t \mid t),
$$

其中𝜋表示一条路径。同样的，以上面可能的路径A - - B为例： 

$$
P(\pi \mid X) = P(A \mid t=1)⋅P(- \mid t=2)⋅P(- \mid t=3)⋅P(B \mid t=4)
$$    
   
**PS**：CTC时间步在常见OCR模型中的含义：  
如果在 OCR 模型中使用 CTC 解码，而模型的结构并非 RNN，而是基于 CNN 等“平行时间步”的结构（如 CRNN 中的卷积部分或完全基于 CNN 的模型），时间步实际上指的是特征图的宽度维度上每个位置的输出。在backbone中经过CNN提取特征后：输出一个特征图，大小可能是 𝐶×𝐻′×𝑊′，其中𝑊′对应于“时间步”，而每个时间步表示特征图**宽度维度**上的某个位置，它对应输入图像中的某一列区域。
    
看到这里，你可能会疑惑：既然能得到每个时间步上预测字符的概率分布，那么直接取每个时间步上概率最大的那个字符组成目标序列就可以了，哪里来的多条路径呢？   
实际上，CTC的核心思想正是**允许多条路径对同一个目标序列的贡献**，来实现对目标序列概率的全面建模（也称**波束搜索解码**）。如果采用**贪心解码**，即在每个时间步𝑡上采用概率最大的字符，最终只能得到一条路径。假如我们有概率分布矩阵：
![例图1](https://ezcodingsen.github.io/images/notebook/CTC decoder in ppocrv4 rec/1.png)
可以计算每条可能路径的概率，而贪心解码只能得到一条路径，并直接得到最终的输出。这样可能陷入**局部最优**的问题。   
### 3.目标序列的概率   
而CTC解码器的关键是计算目标序列𝑌的概率，这通过将所有可能路径的概率相加实现   

$$
P(Y \mid X) = \sum_{\pi \in \text{Alignments}(Y)} P(\pi \mid X)
$$   
### 4.序列归约   
从对齐路径𝜋到最终序列𝑌，需要去掉所有 "blank" 标签以及**相邻的重复字符**。   
正如上图所描述的，通过累加和归约，我们可以的到不同目标序列（AA，AB，AAB...）的概率,再去判断哪个目标序列的概率最大，作为最终的输出。

CTC decoder在PaddleOCR中的代码实现
------

在ppocrv4 hgnet这个文本识别模型当中，采用的是CTCHead和NRTRHead共用的MultiHead模式。训练时，两个头都发挥作用。在实际推理当中，进行解码的只有CTCHead。  
而在模型结构中，使用CTCHead解码前，还引入了svtr这个Neck将backbone输出的视觉特征进行融合。我们可以从MultiHead的forward函数中看到详细的流程：  

```python
def forward(self, x, targets=None):
    if self.use_pool: # 默认为false
        x = self.pool(
            x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2])
        )
    ctc_encoder = self.ctc_encoder(x) # Neck部分进行特征融合
    ctc_out = self.ctc_head(ctc_encoder, targets) # 对Neck的输出进行CTC解码
    head_out = dict()
    head_out["ctc"] = ctc_out
    head_out["ctc_neck"] = ctc_encoder
    # eval mode
    if not self.training: # 在这里可以看到，在实际推理时，只使用了CTC解码
        return ctc_out
    if self.gtc_head == "sar": # 后面这部分是NRTRHead，这里就不详解了
        sar_out = self.sar_head(x, targets[1:])
        head_out["sar"] = sar_out
    else:
        gtc_out = self.gtc_head(self.before_gtc(x), targets[1:])
        head_out["gtc"] = gtc_out
    return head_out
```
其中的self.ctc_encoder实际是特征融合的Neck部分，其输入是backbone输出的Bx1024x1x40特征图，经过一系列融合之后输出ctc_encoder为Bx40x120的融合特征。（这里就不详细解释Neck部分了）  
   
解释一下为什么是这个形状：  
Bx1024x1x40是HGGNet这个backbone设定的输出形状，1x40是适配条状文本识别的HxW。W也就是序列长度/时间步t的长度（参考上面PS中提到的“平行时间步”，便于理解。）  
Bx40x120中40是序列长度，120是Neck设定的dims（默认为64）  
  
输入CTCHead进行解码的代码，我截取了关键的部分放在下方：  
```python
predicts = self.fc(x)
predicts = F.softmax(predicts, axis=2)
            result = predicts
```
fc是一个全连接层，其输入Bx40x120，通道数为：120（中间有维度转换，后面又转回去了，不重要），输出通道数为：字符表的长度（我这里使用的自己的字符表，总共75个字符，加上自动添加的"blank"，一共76个字符。长度也就是76。如果使用的是ppocrv4默认的字符表的话大概是6600多个字符。）这个全连接层将输入的融合特征图的通道数映射到与字符表一样，方便后面对每个字符的概率进行预测。   
在我的模型中：Bx40x120 --> Bx40x76  
   
然后就是通过softmax预测，将第3维度（76）变为概率嘛，代表字符表中各个字符出现的概率。  
CTCHead的功能到这里就结束了，后续部分在后处理CTCLabelDecode中，关键代码如下：
```python
preds_idx = preds.argmax(axis=2)
preds_prob = preds.max(axis=2)
text = self.decode( # 归约，去重，将字符序号变成字符表中对应的文字，组合后返回文本text
    preds_idx,
    preds_prob,
    is_remove_duplicate=True,
    return_word_box=return_word_box,
)
```
可以看到，ppocrv4采用的CTC解码是上面提到的**贪心解码**，也就是直接取概率最大的字符组成唯一路径，直接得到输出。（汗流浃背了） 
后续我将修改后处理为波束搜索解码，并进行对比测试。












