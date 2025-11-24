---
marp: true
theme: gaia
paginate: true
headingDivider: 3
backgroundColor: white
math: MathJax
transition: fade
style: |
  /* Define the style of "morph" class */
  .morph {
    display: inline-block;
    view-transition-name: var(--morph-name);
    contain: layout;
    vertical-align: top;
  }

auther: Pingzhi Tang
---
<style>
img {
    display: block;
    margin: auto;
}
</style>


<!-- _class: lead -->
# <!-- fit --><span class="morph" style="--morph-name:quant;">Quantization</span>
#### Fundamentals, QAT and more.

> Quantization is set of techniques to reduce the precision, make the model smaller and training faster in deep learning models. 

Pingzhi Tang
stanleytang@stu.pku.edu.cn
<!-- footer: Huggingface [blog](https://huggingface.co/blog/merve/quantization) -->

<!-- 
这里引用了 huggingface 上一篇 blog 对量化的定义
上个学期 qb 讲过一次量化，但是上次主要关注的是 PTQ 也就是 post-training quantization
这种量化方法的好处是对算力要求比较少，但是效果有上限，当 bitwidth 非常小的时候效果一般
今天我们先简单的过一下量化的背景知识，然后就重点讲一下 qat 以及一些相关的文章，文章跨度从 20年到上个月都有
-->

$$
\DeclareMathOperator{\round}{round}
\DeclareMathOperator{\if}{if}
\DeclareMathOperator{\quantize}{quantize}
\DeclareMathOperator{\dequantize}{dequantize}
\DeclareMathOperator{\clip}{clip}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\E}{\mathbb{E}}
$$


# Table of Contents
<style scoped>
li {
   font-size: 1.2rem;
}
</style>
- <span class="morph" style="--morph-name:quant-fund;"><span class="morph" style="--morph-name:quant;">Quantization</span> Fundamentals</span>
- Determining Quantizer Parameters
- Quantization-Aware Training (QAT)
- Evaluating Quantized LLMs
<!-- footer: \b -->

# <span class="morph" style="--morph-name:quant-fund;"><span class="morph" style="--morph-name:quant;">Quantization</span> Fundamentals</span>
<!-- _class: lead -->
<!-- footer: Gong et al., [[arxiv]](https://arxiv.org/abs/2409.16694) -->

## <span class="morph" style="--morph-name:quant;">Quantization</span> of LLMs: a Taxnomic Review
Based on targets:
- weight-only
- weight & activation
- KV cache

Those different types have different mechanisms to archieve real acceleration and storage saving (or they don't acually?).

### Aside: Data Transmission during Inference
![bg w:1100](images/e126ee3f4c860bd3318faee593c24d8c17b10c87467b626a0a6b3f3bc0406a0e.png)  

<!-- 
要理解量化是怎么加速模型推理的，我们要来看看 gpu 的存储体系结构
首先是gpu 之外的存储，就是主存（内存）
然后 gpu 上有 On-chip caches (L2 cache, shared memory, and registers)和off-chip caches (device memory or global memory, host memory)
on-chip 快但是容量小
所以在llm 推理的时候，需要先把 data 一小块一小块地 load 逐步的 load 到 on-chip 的存储上
-->
<!--
Host memory → Device memory. 只有 weight 需要这一步，activation 是实时产生的，不需要从内存传输。这一步是很慢的，用量化之后的 weight，比较 compact，这一步的传输速率就快了
Off-chip memory → On-chip memory. 在推理过程中，我们从离芯片较远的全局内存（off-chip global memory）复制一块权重和激活值到芯片上的 L2 缓存和共享内存，以便进行矩阵乘法计算（MatMul）。数据传输的数量通常由矩阵乘法（MatMul）内核的设计决定，通常是 SIMT 执行的元素数量的整数倍。在 A100 GPU 上，这一过程的带宽为 1555 GB/s。
Shared memory → Registers. 为了加快计算，量化/反量化操作以及矩阵乘法 (MatMul) 始终在寄存器中进行。因此，我们需要将权重和激活值从共享内存以小块形式复制到寄存器中。该过程的带宽为 19400 GB/s，比 PCIe 高出 10 倍以上，并且计算强度仅为其 1/780。
Offloading (Registers → shared memory → off-chip memory). 计算结果被复制或累加到共享内存中的相应元素。在计算完当前数据块后，共享内存中的结果被卸载到离芯片较远的全局内存（off-chip memory）。在处理下一个数据块之前，存储前一个块权重和激活值的内存可以被释放，以优化内存使用。
-->
<!-- 
还有一个需要注意的是，一般 gpu 在传输数据的时候是需要按照固定的位大小的（32 位），所以虽然量化的时候可以量化到任意的位数，但是还是需要bit-packing，使得memory bandwidth 不被浪费
 -->

### Weight-only VS. W&A Quantization
![bg w:900](images/7d64b2767fea4bdd4e4dbbdc3b892647b638d58f0019cd9ab4f7232a7ef9ff1b.png)  
<!--  
第一个是准备阶段，和推理的时间无关
中间的是 weight-only，可见三种颜色分别表示加速，变慢和不变
所以我们可以看到一般来说 w&a 加速要大于 w-only，实际上，一般的推理引擎下，w-only 几乎是不怎么加速的
-->
<!-- 
The extra operations are the quantization for activation from FP16 to low-bit integer before MatMul, and the datatype casting for the results from INT32 to FP16 after MatMul. Weight & activation quantization yields greater acceleration compared to the weight-only quantization because the computationally intensive MatMul usually can be accelerated by lower bitwidth kernels, which use more efficient instructions and a better degree of parallelism.
-->
<!-- 
进一步加速推理有两种设计思路
(1) Faster Quantization and Dequantization (or datatype conversion).
(2) Faster MatMul Kernel. GEMV can be more flexible and efficient in fitting various bitwidths than GEMM, and even receives input matrices with two bitwidths
-->


### Weight-only VS. W&A <span class="morph" style="--morph-name:quant;">Quantization</span>
- The `MatMul` operation is conducted under high precision in weight-only quantization, so it theoretically supports **non-uniform quantization with arbitrary bitwidth**. (see NF4 in [QLoRA](https://arxiv.org/abs/2305.14314), learnable thresholds in [N2UQ](https://arxiv.org/abs/2111.14826))
- W&A quantization requires customized computation kernels with assembled GEMM/GEMV instructions, especially when the quantization theme is fine-grained.

<!-- 
一但我们摆烂决定用weight-only高精度的矩阵乘法之后，weight 怎么量化就都可以了
理论上你可以用任何你觉得可以减少量化误差的方法
但是当然推理的过程中解量化也需要时间
-->


### KV Cache <span class="morph" style="--morph-name:quant;">Quantization</span>
![picture 3](images/ecaa4f0843c3621cbd0ed7be6150a0d6b1c634fbb905c08cb065fa8bb742b0d0.png) 
<!-- 
还有一种就是 kv-cache 量化了
基本形式非常简单，就是把 kv-cache 的 activation 计算出来之后直接进行量化，然后 concat 在kv-cache 上
这样就一方面减少了显存的占用，也使得推理的时候 move kv-cache 的开销减少
当然也可以用量化之后的 kv-cache直接做低精度 attention，只是这里画的是解量化的方法
When the cache size exceeds its limit, the earliest key-value pairs will be dropped.
Then we dequantize the matrices to FP16 before conducting multi-head attention forward propagation with the newly generated query Qnew.
Compared to the FP KV cache, the quantized one occupies less storage in device memory and spares less time in KV data transmission in the caching system due to the smaller data bytes.
-->


### KV Cache <span class="morph" style="--morph-name:quant;">Quantization</span>
- Similar to weight-only algorithms, the quantized key-value pairs usually need to be dequantized to floating-point before MatMul, otherwise, the **specific system support** of multiplying low-bit to floating-point is required.
* Details omitted, since the similar topic was discussed last week.

### Bonus: <span class="morph" style="--morph-name:quant;">Quantization</span> During Training
Used in training, e.g. DeepSeek-V3
![w:1000](images/7319ad6e491b9e0f80825728f78fbf9538d21cc893594a73ba046628a98f521e.png)  
<!-- _footer: DeepSeek-AI et al., [[arxiv]](https://arxiv.org/abs/2412.19437) -->
<!-- 
上述说到的都是量化在推理的时候的好处, 现在额外讲一下训练的时候的好处,这里是 deepseek-v3在训练过程中用到的量化
其实大概的思路也是一样的，就是通过这个 FP8 的乘法 kernel 加速训练，并减少计算量
但是之前的 fp8 训练会被 outliers 限制，deepseek 这里使用了非常精细的量化粒度，然后设计了对应的非常高效的乘法运算核
这样的细粒度量化也使得他们可以把所有的fp8 都是用 E5M2，因为一个 group 中的数据 scale 之后在大小上已经相差不大了
他们还仔细的选择了到底量化哪些东西，比如在图上可以看到master weights, weight gradients, and optimizer states都是高精度的；他们还选择了the embedding module, the output head, MoE gating modules, normalization operators, and attention operators保持高精度
While low-precision training holds great promise, it is often limited by the presence of outliers in activations, weights, and gradients. 
a fine-grained quantization strategy: tile-wise grouping with 1 × N_c elements or block-wise grouping with N_c × N_c elements.The associated dequantization overhead is largely mitigated under our increased-precision accumulation process, a critical aspect for achieving accurate FP8 General Matrix Multiplication (GEMM). 
cache and dispatch activations in FP8, while storing low-precision optimizer states in BF16.
most compute-density operations are conducted in FP8, These GEMM operations accept FP8 tensors as inputs and produce outputs in BF16 or FP32
certain operators still require a higher precision due to their sensitivity to low-precision computations. Besides, some low-cost operators can also utilize a higher precision with a negligible overhead to the overall training cost.
    for stable training dynamics: the embedding module, the output head, MoE gating modules, normalization operators, and attention operators.
    guarantee numerical stability: master weights, weight gradients, and optimizer states
In contrast to the hybrid FP8 format adopted by prior work, which uses E4M3 (4-bit exponent and 3-bit mantissa) in Fprop and E5M2 (5-bit exponent and 2-bit mantissa) in Dgrad and Wgrad, we adopt the E4M3 format on all tensors for higher precision. We attribute the feasibility of this approach to our fine-grained quantization strategy, i.e., tile and block-wise scaling. By operating on smaller element groups, our methodology effectively shares exponent bits among these grouped elements, mitigating the impact of the limited dynamic range.
-->

## <span class="morph" style="--morph-name:quantizer;">Quantizer</span> Design
- Uniform quantizer
    - Affine (asymmetric)
    - Scale (symmetric)
    * Stochastic
- Nonuniform quantizer
    - NF4, etc.
![bg right:50% w:600](https://www.researchgate.net/profile/Biao_Sun3/publication/309083844/figure/download/fig1/AS:416736198316035@1476369060881/a-Mid-rise-uniform-quantization-The-mid-point-value-within-a-cell-is-taken-as-the.png)  
<!-- footer: Wu et al., [[arxiv](https://arxiv.org/abs/2004.09602)] -->
<!-- 
可想而知, 表达精度: nonuniform > affine > scale.
nonuniform 硬件支持不好, 但是对于 weight only 完全可用.
stochastic: bonus
-->

### Affine <span class="morph" style="--morph-name:quantizer;">Quantizer</span>
For a given parameter $x \in \mathbb{R}$, the corresponding quantized version $x_q \in \{-2^{b - 1}, -2^{b - 1} + 1, ..., 2^{b - 1} - 1\}$ is given by:
$$
\begin{aligned}
x_q &= \text{quantize}(x, b, s, z) \\
&= \clip(\text{round}(s \cdot x + z), -2^{b - 1}, 2^{b - 1} - 1)
\end{aligned}
$$
And the corresponding dequantize function:
$$
\hat{x} = \text{dequantize}(x_q, s, z) = \frac{1}{s}(x_q - z)
$$
<!--  
伸缩变形之后 round, z 即是零对应的量化值.
实际上这里也不一定是$x_q \in \{-2^{b - 1}, -2^{b - 1} + 1, ..., 2^{b - 1} - 1\}$，我们也可以选择量化之后的数据类型为无符号型
-->

### Affine <span class="morph" style="--morph-name:quantizer;">Quantizer</span>: A Closer Look at Quant. Param.
Affine transformation function, $f(x) = s \cdot x + z$ :
$$ s = \frac{2^b - 1}{\alpha - \beta} $$
$$ z = -\text{round}(\beta \cdot s + 2^{b - 1}) $$
- Representable range $[\beta, \alpha]$ (approximately).
- $z$ is rounded to an integer value so that the real value of zero is exactly representable.
- How to select $\alpha, \beta$ ?
<!-- 
z被四舍五入到整数, 使得 0 可以被准确的表示: padding 等.
alpha, beta 的选取: 大学问, 说到底所有的 PTQ 方法的不同就是如何确定这两个参数
-->

### Scale <span class="morph" style="--morph-name:quantizer;">Quantizer</span>
Pretty much the same, but without $z$, making the quantization symmetric.
$$ x_q = \quantize(x, b, s) = \clip(\round(s \cdot x), -2^{b - 1} + 1, 2^{b - 1} - 1) $$
$$ \hat{x} = \text{dequantize}(x_q, s) = \frac{1}{s}x_q $$
- The Representable range $[-\alpha, \alpha]$ is symmetric.
- $-2^{b-1}$ is abandoned for symmetry and simplicity.
<!-- 
scale量化和 affine 量化的区别, 很明显就是开销上, 无论是显存的开销还是运算的开销
之后可以看到, affine 量化的低精度乘法操作开销比 scale 大, 很多时候在精度要求不高时就使用 scale量化了.

这里我们也埋了一个小小的伏笔，虽然我们这里写了把 -2^{b-1} 舍去了，这个对 bitwidth 比较大的情况确实是有好处的，但是对于 bitwidth 很小的时候显然是不能这么做的（比如 2bit）
所以上个月 meta 发了一个文章也对这个问题做了一点讨论，我们之后也会提到
-->

### Bonus: Stochastic <span class="morph" style="--morph-name:quantizer;">Quantizer</span>
Stochastic quantization models the quantizer as an additive noise, followed by rounding. The stochastic quantizer is given by:
$$ x_q = \clip(\text{round}(s \cdot x + z + \varepsilon), -2^{b - 1}, 2^{b - 1} - 1), $$
$$ \varepsilon \sim \text{Uniform}(-0.5, 0.5) $$
- <span class="morph" style="--morph-name:stochastic;">In expectation, the stochastic quantizer reduces to a **pass-through of the floating point weight, with saturation for values outside the range**. </span>
- Most inference hardware does not support it :(.
<!-- _footer: Krishnamoorthi et al., [[arxiv]](https://arxiv.org/abs/1806.08342) -->
<!-- 
这个随机量化的思想很有趣, 实际上也是之后会提到的 ste 的思想.
量化操作相当于引入一个随机的误差, 这个误差可以建模为一个加性的误差
在下面一页我们可以看到, 这个随机量化在期望上就相当于 x 直通
这个性质非常好, 因为一方面他保证了量化在期望上是没有损失的; 另一方面使得通过round 函数之后的反向传播变得非常合理
但是这个方法并没有被主流推理系统实现, 可能是因为取随机数这个操作是很慢的.
-->

### Bonus: Stochastic <span class="morph" style="--morph-name:quantizer;">Quantizer
- <span class="morph" style="--morph-name:stochastic;">In expectation, the stochastic quantizer reduces to a **pass-through of the floating point weight, with saturation for values outside the range**. </span>
- Assume $x$ in range $[a, a + 1]$, where $a \in \Z$. We have:
$$
\begin{align} 
\E(\round(x + \varepsilon)) &= a \cdot \Pr(\round(x + \varepsilon) = a) \\
&+ (a + 1) \cdot \Pr(\round(x + \varepsilon) = a + 1) \\
&= a \cdot (a + \frac{1}{2} - x + \frac{1}{2}) + (a + 1) \cdot (\frac{1}{2} - a - \frac{1}{2} + x) \\
&= x
\end{align}
$$
<!-- _footer: Krishnamoorthi et al., [[arxiv]](https://arxiv.org/abs/1806.08342) -->
<!-- 
这里简单推导了一下, 就是和两边整数的距离
-->

### Nonuniform <span class="morph" style="--morph-name:quantizer;">Quantizer</span>
$$x_q = q_i, \quad \if x \in [\Delta_i, \Delta_{i + 1}]$$

Adjusting the quantization resolution according to the density of real-valued distribution.
- **Quantile Quantization**: to find an optimal data type that ensures each quantization bin has an equal number of values assigned from the input tensor. 
- **`NF4` data type**: assume pretrained weights to be zero-centered normal distribution with standard deviation $\sigma$.
<!-- _footer: Dettmers et al., [[arxiv]](https://arxiv.org/abs/2110.02861); Dettmers et al., [[arxiv]](https://arxiv.org/abs/2305.14314) -->
<!-- 
qi: the discrete quantization levels
∆i: the quantization steps (thresholds)
Note that neither qi’s nor ∆i’s are uniformly spaced.
nonuniform 量化一般的效果都比 uniform 的好, 但是推理的时候解量化的开销比较大
怎么得到量化levels?: 一般来说就是画出数据分布的曲线, 然后使得在不同 level之间的数据比例相同
-->

### Nonuniform Quantizer
- Since the output of the nonuniform quantization are floating-point weights and activations, their multiplication can no longer be directly conducted under integer arithmetic.

We will come to this later!
<!-- _footer: Liu et.al., [[arxiv]](https://arxiv.org/abs/2111.14826) -->
<!-- 
但是他的损失较小使得他可以很好的利用在 weight-only quantization 里面
-->


## Quantization Granularity
- Per tensor
- Per row / column (token, channel...)
- Per group
- Per element

![w:900](images/dbb0b6cf37903e4aedc7a25c79056cf6a4c1cc71bf21825466a9a73feec3ed3c.png)  

## Quantization Granularity

<style scoped>
li {
   font-size: 0.9rem;
}
p {
   font-size: 0.9rem;
}
</style>

Three factors to consider when choosing granularity: **model accuracy**, **memory consumption** and **computational cost**.
- For W&A quantization, there's also a hard constraint from GEMM.
- Consider a linear layer that performs a matrix multiplication
$$Y = XW$$
where $X = (x_{ik}) \in \R^{m\times p}$, $W = (w_{kj} ) \in \R^{p\times n}$.
- The corresponding quantized version: $X_q = (x_{q, ik}) \in \Z^{m\times p}, \quad W_q = (w_{q, kj} ) \in \Z^{p\times n}$

<!-- 
这里讲的是不额外自己设计 kernel 的情况下
-->

## Quantization Granularity: Scale Quant.

<style scoped>
p {
   font-size: 0.9rem;
}
</style>

First, consider tensors quantized at **the finest granularity, per-element**, with scale quantization:
$$
y_{ij} = \sum_{k=1}^{p}{x_{ik} \cdot w_{kj}} \approx 
\sum_{k=1}^{p}{\dequantize(x_{q,ik}, s_{q, ik}) \cdot \dequantize(w_{q,kj}, s_{w,kj})} = \sum_{k=1}^{p}{\frac{1}{s_{x,ik}}x_{q,ik} \cdot \frac{1}{s_{w,kj}}w_{q,kj} }
$$
In order to use integer matrix multiplication the scales must be **factored out** of the summation, for which the scales must be independent of k:
$$
\frac{1}{s_{x,i} \cdot s_{w,j}}\sum_{k=1}^{p}{ x_{q,ik} \cdot w_{q,kj}}
$$

## Quantization Granularity: Scale Quant.
$$
\frac{1}{s_{x,i} \cdot s_{w,j}}\sum_{k=1}^{p}{ x_{q,ik} \cdot w_{q,kj}}
$$

- Integer matrix multiplication is possible as long as the quantization granularity is **per-row (or coarser) for activations** and **per-column (or coarser) for weights**, if no customized computation kernels designed.
- For activations, per-row (per-token) quantization will lead to the need of **Dynamic Quantization**.

<!-- 
对于 llm 来说，对于 activation 的 per-token量化是必须on-the-fly 进行的，因为我们不能提前确定每个token 的量化参数
在 cnn 中一般就直接用 per-tensor 了，也够用了
-->


## Quantization Granularity: Affine Quant.

<style scoped>
p {
   font-size: 0.8rem;
}
li {
   font-size: 0.8rem;
}
</style>

For affine quantization:
$$
\begin{align}
y_{ij} & \approx \sum_{k=1}^{p}{
      \frac{1}{s_{x,i}}(x_{q,ik}-z_{x,i})\frac{1}{s_{w,j}}(w_{q,kj}-z_{w,j})}   \\
    & = \frac{1}{s_{x,i}s_{w,j}} \bigg(
      \underset{(1)}{\sum_{k=1}^{p}{x_{q,ik} w_{q,kj}}}
    - \underset{(2)}{\sum_{k=1}^{p}{(w_{q,kj} z_{x,i} + z_{x,i} z_{w,j})}}
    - \underset{(3)}{\sum_{k=1}^{p}{x_{q,ik} z_{w,j}}}
    \bigg)
\end{align}
$$
1) The first term is the integer dot product, just as in scale quantization. 
2) The second term consists of only integer weights and zero-points. 
3) The third term, however, involves the quantized input matrix $X_q$, and thus cannot be computed offline. 

<!-- 2. As a result, this term can be computed offline, only adding an element-wise addition at inference time. If the layer has a bias then this term can be folded in without increasing inference cost.  -->
<!-- 3. This extra computation, depending on implementation, can introduce considerable overhead, reducing or even eliminating the throughput advantage that integer math pipelines have over
reduced precision floating-point. -->
<!-- Note that this extra computation is incurred only if affine quantization is used for the weight matrix. Thus, to maximize inference performance we recommend using scale quantization for weights. weight 基本上是 0 均值的 gauss! -->


## Quantization Granularity: Dynamic Quant.
- Per-token quantization for LLMs? Gonna need **dynamic quant**.
    - Determining quant params on the fly.
![w:900](images/1f042f16a48b4bad32875de618d4e06bb5be1b9a482190e0943f2abd8c8299b0.png)  
<!-- _footer: Gong et al., [[arxiv]](https://arxiv.org/abs/2409.16694) -->

<!-- 
由前面的叙述可以知道, 如果要对 activation 进行 per-row 的量化, 需要动态的确定量化参数, 这就是发生在大模型上的事.
对于 CNN 来说, per-tensor 的量化已经足够, 不需要动态量化了(动态量化会带来运行性能上的损失)
-->


## Modeling Simulated Quantization in the Backward Pass
For quantization-aware training, we model the effect of quantization using **simulated quantization** operations, i.e,
$$ \hat{x} = \dequantize(\quantize(x, \theta_{\text{quant}}), \theta_{\text{quant}}) $$
**One challenge**: quantization operation’s derivative is undefined at the step boundaries and zero everywhere else.
* Solution: <span class="morph" style="--morph-name:ste_full;">Straight-through Estimator (STE)</span>

<!-- 我们有时候会用到经过量化的步骤之后的梯度, 这时候就需要用到梯度的近似估计, 因为 round函数是没有梯度的 -->



### <span class="morph" style="--morph-name:ste_full;">Straight-through Estimator</span></br>(<span class="morph" style="--morph-name:ste;">STE</span>)
STE approximates the derivative of the fake quantization function to be $1$ for inputs in the representable range $[\beta, \alpha]$.
$$ \frac{\mathrm{d}\hat{x}}{\mathrm{d}x} = 
\begin{cases}
0, & y < \beta \ \text{or} \ y > \alpha \\
1, & \beta \le y \le \alpha
\end{cases}
$$
![bg right:40% w:500](images/48f53925f83df26430f1dca0a04aedfc635e94addf61a6ab837dce2f28d41b75.png)  

<!-- 
这个近似其实非常好理解，就是把阶梯型的函数当成identity 函数，得到梯度就是 1.
接下来我们来看看一种更加有趣的理解，这种理解可以给我们一些新的 insight。
-->


### <span class="morph" style="--morph-name:ste;">STE</span>: A Stochastic Interpretation
Stochastic binarization & deterministic binarization
- In stochastic binarization, real-valued variable $x^r$ is binarized to $-1/1$ stochastically according to its distance to $-1/1$:
$$
\begin{split}
\widetilde{x}^{b} = \left\{
             \begin{array}{lr}
             \ \!-1,  \!\! & \text{with probability} \ \ p = {\clip}( \frac{1-x^r}{2}, 0, 1) \\
             \quad\!1,  \!\! & \text{with probability} \ \ p = {\clip}( \frac{1+x^r}{2}, 0, 1)
             \end{array}
\right.
\end{split}
$$
where $\widetilde{x}^b$ denotes the stochastic binary variables.
<!-- footer: Liu et al., [[arxiv]](https://arxiv.org/abs/2111.14826) -->

<!-- 
注意我们用带 tilde 的表示随机二值，不带的表示确定二值

随机二值和确定性二值函数是在二值神经网络里提出的概念, 我们用在这里有助于我们对 ste 的理解
随机二值就是和我们之前推导 stochastic quantizer 里差不多
-->

### STE: A Stochastic Interpretation
- Expected gradient:
$$
\begin{align*}
\frac{\partial x^b}{\partial x^r} &\approx \mathbb{E}_{\widetilde{x}^b}[\frac{\partial \widetilde{x}^b}{\partial x^r}] = \frac{\partial}{\partial x^r} \mathbb{E}[\widetilde{x}^b] \\
& = \frac{\partial}{\partial x^r} (-1 \times p_{\{\widetilde{x}^b = -1\}} + 1 \times p_{\{\widetilde{x}^b = 1\}}) \\
& = \frac{\partial}{\partial x^r} {\clip}(x^r,-1,1) \\
\end{align*}
$$
* This arrives at the common straight-through estimator for the binarization functions.

<!-- 
我们考虑随机二值的梯度的期望  
在这里我们得到了, 随机二值的ste
-->


### STE: A Stochastic Interpretation
- Meanwhile, deterministic binarization function in forward pass:
$$
\begin{split}
x^b = \left\{  
             \begin{array}{lr} 
             \ -1, &   p_{\{\widetilde{x}^b = -1\}} > 0.5 \\
             \quad \ \  1, &  p_{\{\widetilde{x}^b = -1\}} \leqslant 0.5 
             \end{array}  
\right.  = \left\{  
             \begin{array}{lr}  
             \  -1, & \!x^r < 0 \\ 
             \quad \ \ 1, & \! x^r \geqslant 0  
             \end{array}  
\right.
\end{split}
$$
* To this end, we show that STE encodes the expectation of stochastic binarization in the backward approximation to the forward deterministic binarization functions. 

<!-- 
为此，我们证明了 STE 在对前向确定性二值化函数进行后向逼近时，编码了随机二值化的期望。
也就是说，STE 估计了这样一个没有梯度的确定性二值函数的反向过程。
正向：确定性二值
反向：随机二值
-->



### Generalizing STE: G-STE
**Lemma**: A quantization function $x^q$ can be regarded as a summation of deterministic binarization functions $x^{b_{\{0,1\}}}$ with different thresholds: $x^q = x^{b_{\{0,1\}}}_{1} + x^{b_{\{0,1\}}}_{2} + ... + x^{b_{\{0,1\}}}_{n}$.
where 
$$
x^{b_{\{0,1\}}}_{i} = \left\{  
            \begin{array}{lr} 
            \quad 0, &   x^r < thre_i \\
            \quad 1, &   x^r \ge thre_i
            \end{array}  
\right. 
$$
![bg right:40% w:550](images/95c4be77274a5eaf738c4fe8f7fd019eb87b3d0f2a8ab8c04d3111fd74412a63.png)  

<!-- 
前面说明了一个uniform 量化的反向传播，但是对于 non-uniform 量化，情况就没有这么简单了

我们现在再来回头看一个 non-uniform 的量化函数，其实就是一个分段函数
那么就可以理解为多个不同 threshold 的 0/1 二值函数的和
-->


### Generalizing STE: G-STE
<style scoped>
p {
   font-size: 0.9rem;
}
</style>
Consider the first binarization, in the range of $[s, s+a_1]$:
$$
\begin{split}
\widetilde{x}^{b_{\{0,1\}}}_{1} = \left\{  
             \begin{array}{lr}
             \!0 \ \ \ \text{with probability} \ p\!=\!{\clip}(\frac{s + a_1 - x^r}{a_1}, 0, 1) \\
             \!1 \ \ \ \text{with probability} \ p\!=\!{\clip}( \frac{x^r-s}{a_1}, 0, 1)
             \end{array}  
\right. 
\end{split}
$$
Similarly, we can derive the expected gradient: 
$$
\begin{align*}
\frac{\partial x^{b_{\{0,1\}}}_{1}}{\partial x^r} &= \mathbb{E}[\frac{\partial \widetilde{x}^{b_{\{0,1\}}}_{1}}{\partial x^r}] =
\frac{\partial}{\partial x^r} \mathbb{E}[\widetilde{x}^{b_{\{0,1\}}}_{1}]  \\&= \frac{\partial}{\partial x^r} (0 \times p_{\{\widetilde{x}^q = 0\}} + 1 \times p_{\{\widetilde{x}^q = 1\}}) = \frac{\partial {\clip}(\frac{x^r-s}{a_1}, 0, 1)}{\partial x^r}
\end{align*}
$$

<!--
前一页中的都是确定性的二值函数，现在开始考虑随机二值，对应着反向传播
实数被随机地量化为$0/1$ with the probability proportional to their distance to $s/s+a_1$
-->

### Generalizing STE: G-STE
Therefore we have **G-STE**:
$$
\begin{align*}
\frac{\partial x^q}{\partial x^r} &= \mathbb{E}[\frac{\partial \widetilde{x}^q}{\partial x^r}] = 
\frac{\partial}{\partial x^r} \mathbb{E}[\widetilde{x}^q] =
\sum_i \frac{\partial}{\partial x_r}\mathbb{E}[ \widetilde{x}_i^{b_{\{0,1\}}}] \\
&= \sum_i \frac{\partial {\clip}(\frac{x^r-d_{i-1}}{a_i}, 0, 1)}{\partial x^r}
= \left\{
        \begin{array}{c}
        \frac{1}{a_i} &  d_{i-1} \leqslant x^r < d_i \\
        0 &  \text{otherwise}
        \end{array}  
\right.
\end{align*}
$$
where $d_0 = s, \ \ d_i = s + \sum_{j=1}^{i} a_j, \ \ i\in\{1,...,2^n -1\}$

<!-- 
利用和刚刚的分析几乎一样的推导方式，我们可以得到整个定义域上的梯度
这就是 G-STE
-->

### Generalizing STE: G-STE
Utilizing G-STE, we can even make the thresholds trainable(!).
$$
\begin{split}
\frac{\partial x^q}{\partial a_i} &= \frac{\partial\mathbb{E}[\widetilde{x}^q]}{\partial a_i} = \sum_j \frac{\partial {\clip}(\frac{x^r-d_{j-1}}{a_j}, 0, 1)}{\partial a_i} \\
&=
\left\{  
        \begin{array}{lr}
        - \frac{x^r-d_{i-1}}{a_i^2} &  d_{i-1} \leqslant x^r < d_i \\
        - \frac{1}{a_j}  & d_{j-1} \leqslant x^r < d_j, {j \ge i+1}\\
        0 & otherwise \\
        \end{array}  
\right.
\end{split}
$$
<!-- 
这个式子的意义就在于：我们可以通过学习，得到模型non-uniform 量化的最佳量化 levels
想象这样一个场景，我们在一个类似推理时的分布上学习量化参数，可以使得在真正推理的时候量化损失较小。
-->

# <span class="morph" style="--morph-name:deter;">Determining Quantizer Parameters</span>
<!-- _class: lead -->
<!-- footer: Wu et al., [[arxiv](https://arxiv.org/abs/2004.09602)] -->

<!-- 
之前我们介绍了几种quantizer，现在我们来讲讲怎么确定对应的量化参数
这个其实是很大的学问，是精度和动态范围的 tradeoff
前面讲的 G-STE 的方法也可以说是一种确定 non-uniform 量化的参数的方法
下面我们来看看其他的主要是 uniform 量化的一些方法
-->

## <span class="morph" style="--morph-name:deter;">Determining Quantizer Parameters</span>: Calibration
How to choose the quantizer parameters(say $s, z$ in uniform affine quantization for activations)?
- <span class="morph" style="--morph-name:statical;">Statistics-based methods:</span>
    - Max: Use the maximum absolute value seen during calibration.
    - Entropy: Use **KL divergence** to minimize information loss between $x$ and $\hat{x}$.

<!-- 
这里主要是和activation的量化有关，activation的分布和推理时有关，所以需要一个calibration set，来让我们确定activation的大致分布
对于 weight 来说，weight 本身是不变的，所以可以直接通过统计的方法提前确定（一般来说就是直接用max）

max 方法就是 prefer 动态范围，牺牲精度
entorpy 就是对量化参数做 grid search，找到使得量化前后 activation 总体分布 kl 散度最小的量化参数
-->

## Determining Quantizer Parameters: Calibration
- <span class="morph" style="--morph-name:statical;">Statistics-based methods:</span>
    - Percentile: Set the range to a percentile of the distribution of absolute values seen during calibration.
    - Exponential moving averages (EMA): moving average of the minimum and maximum values across batches to determine the quantizer parameters. (used in the [original QAT paper](https://arxiv.org/abs/1712.05877) for activation quantization)

<!-- 
percentile: 99% / 99.99%
EMA: 对 activation 值的最大最小做 moving average，用在原始的 qat 论文中
-->

## Determining Quantizer Parameters: Calibration
- Learnable ways:
    - PACT
    - LSQ / LSQ+
    - SEQ

Note: These methods are almost always combined with QAT.
<!-- 
前面几种都是统计的方法，我这里其实比较想讲的是 learnable 的方法
就是通过学习来学到对应的量化参数
-->

### PACT: Parameterized Clipping Activation
Consider a CNN with `ReLU` acted as the layer ActFn.
- The output of the `ReLU` function is unbounded, therefore the quantization after `ReLU` requires a high dynamic range.
- It has been shown that this dynamic range problem can be alleviated by using a clipping activation function, which places an upper-bound $\alpha$ on the output of `ReLU`.
$$
y = \text{PACT}(x) = \left\{\begin{matrix}
0, & x\in(-\infty, 0) \\
x, & x \in [0, \alpha) \\
\alpha, & x \in [\alpha, +\infty)
\end{matrix}\right.
$$
<!-- footer: Choi et al., [[arxiv]](https://arxiv.org/abs/1805.06085) -->

<!-- 
relu 的问题就是没有上界
量化的时候需要确定一个上界，其实就是量化参数中的 s
问题就到了怎么用学习的方法确定这个上界 alpha
-->

### PACT: Parameterized Clipping Activation
After quantization:
$$
y^q = \round(y \cdot \frac{2^k - 1}{\alpha})\cdot \frac{\alpha}{2^k-1}
$$
The corresponding gradient wrt. $\alpha$ (using **STE**):
$$
\frac{\partial y^q}{\partial\alpha} = \frac{\partial y^q}{\partial y} \cdot \frac{\partial y}{\partial\alpha} = \left\{\begin{matrix}
0, & x\in(-\infty, \alpha) \\
1, & x \in [\alpha, +\infty)
\end{matrix}\right.
$$
<!-- 
clip之后做量化，注意 relu 之后的激活值都是正的，所以我们量化也量化到一个无符号数
可以用ste推导出梯度($\frac{\partial y^q}{\partial y} = 1$)
这个时候我们就可以学习出这个 alpha，注意这里是一个 CNN，他是从头训，从头就考虑了量化误差
-->

### PACT: Parameterized Clipping Activation
To avoid large quantization errors due to a wide dynamic range, a L2-regularizer is introduced for $\alpha$ in the loss function.
![w:1000](images/73a55f456b4b17efb0b3135ab8346d2dc9fe3ccb748f2f1af1517b0525113b60.png)  
Become a common practice for CNN quantization!
<!-- 
这个图表示了从头训一个七层 CNN，调整一个 layer 的 alpha，其他的固定无穷大
第一幅图表示除了第一层和最后一层，无穷大的上限是有利的，但是不同层的最小的最佳 alpha 不一样（需要学习）
第二副图考虑了量化之后的结果，可以发现 alpha 较大的时候误差很大，因为量化的精度降低了
第三幅图是加上了 l2 regular 的结果，可以发现局部最低点附近比较陡峭，比较好学习
-->

### LSQ / LSQ+: Learned Step Size Quantization
Consider affine quantization:
$$
\begin{align}
    \bar{x} &= \round \left( \clip\left(\frac{x-\beta}{s}, n, p\right)\right) \\
    \hat{x} &= \bar{x}\cdot s + \beta
\end{align}
$$
To flexibly learn the quantizer parameters $s, \beta$, we need the gradient of $\hat{x}$ wrt. $s, \beta$.
<!-- footer: Esser et al., [[arxiv]](https://arxiv.org/abs/1902.08153); Bhalgat et al., [[arxiv]](https://arxiv.org/abs/2004.09576) --> 

<!--
下面介绍更加直接的学习量化参数的方法，这个两个：LSQ 和 LSQ+基本上是一样的，只是一个是 scale量化一个是 affine 量化，后者多了一个对 zero point 的求导

这里的 n 和 p 分别表示目标数据类型的最小值和最大值

这里的记号和之前的 affine quantizer 的记号有点不太一样，乘和除、加和减是反的
注意这里的\beta是浮点类型，而不是之前所说的整型，这样才能有梯度来学习
-->

### LSQ / LSQ+: Learned Step Size Quantization

<style scoped>
p {
   font-size: 0.9rem;
}
</style>

The gradient update of the parameter $s$ is calculated using:
$$
\begin{align}
    \frac{\partial \hat{x}}{\partial s} &= \frac{\partial \bar{x}}{\partial s} s+\bar{x} \\
    &\approx 
    \left\{\begin{array}{ll}
        -\frac{x-\beta}{s}+ \round({\frac{x-\beta}{s}}) & \text { if } n<\frac{x-\beta}{s}<p \\
        n \text{ or } p & \text { otherwise. }
    \end{array}\right.
\end{align}
$$
Similarly, the gradient update of $\beta$:
$$
\begin{align}
\frac{\partial \hat{x}}{\partial \beta}=\frac{\partial \bar{x}}{\partial \beta} s+1 \approx \left\{\begin{array}{ll}
0 & \text { if } n<\frac{x-\beta}{s}<p \\
1 & \text { otherwise. }
\end{array}\right.
\end{align}
$$


### LSQ / LSQ+: Learned Step Size Quantization
![w:1100](images/f5a0c6ea5c66e18e6f72d4131161845fc81ef7e04d3dd02a9e289c831300808b.png)  
<!-- _footer: Liu et al., [[arxiv]](https://arxiv.org/abs/2502.02631) -->

<!-- 
这是今年二月份 meta 的一篇文章的图
可以看到，在不同的 bit 数上 learnable 结果总是要比统计的好
1.58 = log2(3)
这里的SEQ 可以简单说一下，我们之前的量化策略都是要把 0 用一个整数来表示，但是这样就不可避免的使得整数数据类型里的正数和负数的个数不一样
所以他们对于 bitwidth 很低的情况提出了一种新的量化 level 的分割方法，例如：2bit一般我们会设置为(-2, -1, 0, 1), 他是用(-1.5, -0.5, 0.5, 1.5)
他这里做实验得出结论说对于 3,4bit 以及以上，把 0 表示出来更为重要
但是对于更低的 bit，数据范围的对称性就更为重要了
see more on arxiv
-->

# Quantization-Aware Training (<span class="morph" style="--morph-name:qat;">QAT</span>)
<!-- _class: lead -->
<!-- footer: Jacob et al., [[arxiv]](https://arxiv.org/abs/1712.05877) -->


## <span class="morph" style="--morph-name:qat;">QAT</span>
- Basically, QAT simulates low precision behavior in the forward pass, while the backward pass remains the same. 
- Makes the parameters more robust to quantization!

Recipes:
- Weights need to be quantized before they are multiplied or convolved with the input.
- Outputs of each layer are quantized after the ActFn is applied. 
- `BatchNorm`s need to be folded and `dropout`s to be removed. 

<!-- 
之前的方法（以及之前qb讲的PTQ方法）都是基于预训练好的模型参数，通过一些不同的方法来确定量化参数，或者一些其他的trick来量化
总之是不改变模型本身的参数的，实际上我们希望的就是模型量化前后参数的区别(||w-\hat{w}||)或者输出(||wx - \hat{w}x||)尽可能的小
-->
<!-- 
但是实际上我们不需要拘泥于原始的那个模型，因为我们知道尤其是大模型的参数空间其实是有很多冗余的，完全不一样的参数可能也可以达到相同的效果
所以我们就想能不能通过模拟量化weight和activation的过程，然后训练出一个在推理过程中量化损失比较小的模型
也就是让训练的过程尽可能的模拟推理，所有的 setting 都和推理一致，这就是 QAT 的方法
-->

## QAT
- Insert **simulated quantization** operations into a floating-point network.
$$ \hat{x} = \dequantize(\quantize(x, \theta_{\text{quant}}), \theta_{\text{quant}}) $$
![w:600](images/2efd2956dfe4ce2bb662940519da13ff156a0f4354456d14aad7583cf7a5dcc1.png)  

<!-- 
尽可能的模拟推理过程中的情况
在矩阵乘法计算之前量化 weight，在进入下一层之前量化 activation

注意这里 uint8 * uint8得到的是 uint32，这是因为除了乘还有累加操作（矩阵乘法），所以精度要求需要更高，这个实际上和 cuda kernel 的设计有关

插入伪量化节点之后，在（预训练）数据集上训练 weight 即可
注意这里的反向传播就需要用到之前说的 STE
-->

## <span class="morph" style="--morph-name:qat;">QAT</span>: Why It Works?

![w:1100](images/a773a2709fe5410b8a595ecb6205638e0754cb34f573b07eaf8f7543858ae156.png)  

<!-- 
这里给出了一个 qat 可以起作用的 intuition
左边这个图表示没有使用 qat 的情况，我们学到的最佳 weight 在黄色这个点处；此时我们进行量化，相当于对 weight 进行了一点扰动，这时 w 被舍入到了最近的整数-1，我们可以发现loss 发生了很大的提高。也就是 model收敛到了一个比较 narrow 的局部最优
右边表示用了 qat 的结果，weight 学到了一个更加wide/flat 的局部最优，对于量化的扰动不那么敏感了
-->

## Is <span class="morph" style="--morph-name:qat;">QAT</span> worth the effort?
<style scoped>
img {
    display: block;
    margin: auto;
}
</style>
![w:900](images/6cd6a9a1683bc94451f4de6c73ad4e835eab3f179f4362c2e09ef6b89c347962.png)  
<!-- _footer: Wu et al., [[arxiv](https://arxiv.org/abs/2004.09602)] -->
<!-- 
显然还是很有用的
-->

## <span class="morph" style="--morph-name:qat;">QAT</span> for <span class="morph" style="--morph-name:llm;">LLM</span>s
Problems of QAT for LLMs:
- LLM training is technically difficult and resource intensive. 
- QAT needs training data, which for LLMs is difficult to obtain.
* Here comes <span class="morph" style="--morph-name:llm-qat;">**LLM-QAT**</span> to tackle these problems.
<!-- footer: Liu et al., [[arxiv]](https://arxiv.org/abs/2305.17888) -->

<!-- 
qat 方法是在 cnn 时代提出的，这个方法基本上已经成为 cnn 量化的标准方法，量化到 4bit 或者更低基本上可以没有任何性能损失
但是在 llm 上，qat 有几个问题
第一个就是 llm 本身的训练是多阶段多步的，而且需要很多显卡，所以也是因为这个原因，在 llm 上上次讲的 ptq 方法比较受关注
第二个就是qat 的方法需要预训练数据集，这对大模型来说是不可得的
所以就有了这篇把 qat 用在 llm 上的开山之作，实际上主要还是解决第二个问题
by meta
-->

## <span class="morph" style="--morph-name:llm-qat;">LLM-QAT</span>
- Data-free Distillation
    - Randomize the first token `<start>` from vocabulary and let the full-precision model to complete the sentence.
    - Deterministically selects the top-1 predictions for the first 3~5 tokens and stochastically samples the remaining tokens.
    - Use cross-entropy based logits distillation:
    $$
    \begin{align}
        \mathcal{L}_{CE} = -\frac{1}{n}\sum_c\sum^n_{i=1} p_c^{\mathcal{T}}(X_i)\log(p_c^{\mathcal{S}}(X_i)),
    \end{align}
    $$

<!-- 
1. Furthermore, we discover that the initial few tokens play a crucial role in determining the prediction trend. Therefore, it is important for them to have higher confidence.
2. in the data generation process, it is important to sample the next token
from distribution rather than always selecting the top-1 candidate. By doing so, the next token does not necessarily represent the optimal label for training the student model, as the sampling introduces inherent noise. Consequently, we propose to utilize the predictions from the pre-trained model as soft labels.
-->

## LLM-QAT
- Data-free Distillation

![w:1000](images/52c1b4c8b455cfb3e180995cc7fc1937b78a7b1cd31742108cc78a402810b7b1.png)  

<!-- 
从这里的 ablation study可以发现，直接用 wiki 或者 c4 训泛化性不好
先确定性生成几个 token 再随机生成比较好
-->

## LLM-QAT
- QAT for key-value cache
    - Per-token quantization for key & value
    - During the generation process, the current key and value are quantized, and their corresponding scaling factor is stored.
    - During the training process for QAT, we apply quantization to the entire activation tensors of both the keys and values.

<!-- 
我们在量化的时候会同时量化kv cache，所以在qat的时候也需要考虑到这个部分的量化误差
注意有人可能会说量化的weight和量化的activation相乘不应该产生的就是量化后的kv吗，这里问题在于kv cache的量化位数可能和w&a不一样
-->


## LLM-QAT
![w:900](images/5485057c39ea12f68289a080cad46fd421f2d08fea5d816614979b06e6ac9584.png)  

<!-- 
weight-activation-kvcache
量化到8bit时ptq效果就不错了
但是如果需要进一步降低bitwidth，就需要qat
-->

## EfficientQAT
LLM-QAT's problem:
- Extremely resource-demanding: takes ~500 GPU hours to generate the dataset, and ~900 GPU hours to train a 70B model(!)

EfficientQAT: two-stage strategy
- block-wise training of all parameters (**Block-AP**) 
- end-to-end training of quantization parameters (**E2E-QP**)
<!-- footer: Chen et al., [[arxiv]](https://arxiv.org/abs/2407.11062) -->

<!-- 
QAT 确实有很大的好处，但是在 LLM 上的 qat 需要非常大的资源
首先生成数据的时候需要用原始精度的 model 来蒸馏
其次全量微调整个模型也需要很多 gpu hour
所以去年十月份就有这个来自OpenGV lab 的工作
-->
<!-- 
他把 qat 的过程拆成两个部分
第一步一个 transformer block一个 block 的用 qat
第二步 end to end的训练所有的量化参数
-->

## EfficientQAT
- Block-AP: sequentially conducts QAT within one transformer block before moving on to the next under a block-wise reconstruction framework.
$$
\begin{align}
\arg \min_{\Theta_1, \Theta_2} \lvert\lvert \mathcal{F}(\mathbf{W},\mathbf{X}) - \mathcal{F}\big(Q_w(\mathbf{\mathbf{W}}; \Theta_1,\Theta_2),Q_a(\mathbf{X}, \Theta_2)\big) \rvert\rvert
\end{align}
$$
- E2E-QP: begins with $W_q$ initialized via Block-AP and focuses solely on the training of quantization parameters ($s$ and $z$)
    - Training only $s$ gives similar results as co-training $s$ and $z$.

<!-- 
block-ap: 利用 reconstruction loss 逐层进行 qat，同理利用 ste 估计round 操作的梯度
e2e-qp: 训练所有的 quantization param，注意此时不需要用 ste 了，因为所有的参数已经是量化之后的了，所以前向的时候没有量化操作，只有解量化操作
-->

## EfficientQAT
![w:1000](images/3e3710f1c61e43c301b06d8885cba08b72d97fab7f93e9cc700743646af35f74.png)  

## EfficientQAT
![bg w:430](images/36cde337c7e77c16e254f094b274fbf5777fc4b3ff4d61998fe8500153898f02.png)  
![bg w:600](images/44bae524a9c4a855eb727910776d0dad9aeb946d9736f251e05204c8df9f40e4.png)  

## Efficient<span class="morph" style="--morph-name:qat;">QAT</span>
![w:600](images/d19135c83ebebd85cb3c30509054c2709bdca6ea4a72eb6a8331a08fb583503c.png) 


## Findings on <span class="morph" style="--morph-name:qat;">QAT</span> from ParetoQ
- **Training Budget Allocation**: i.e. Given a fixed training budget (in \#tokens) $\mathcal{B}_{\text{train}} = \mathcal{B}_{\text{FPT}} + \mathcal{B}_{\text{QAT}}$, how should the budget be optimally allocated between full-precision training ($\mathcal{B}_{\text{FPT}}$) and quantization-aware training/fine-tuning ($\mathcal{B}_{\text{QAT}}$)?

| QAT finetuning consistently surpasses both PTQ with $\mathcal{B}_\text{FPT} = \mathcal{B}_\text{train}$ and QAT from scratch with $\mathcal{B}_\text{QAT} = \mathcal{B}_\text{train}$. Optimal performance is nearly achieved by dedicating the majority of the training budget to full precision (FP) training and approximately 10\% to QAT. |
|:-|
<!-- footer: Liu et al., [[arxiv]](https://arxiv.org/abs/2502.02631) -->

<!-- 
9成FP训练+1 成 QAT 最好
-->


## Findings on QAT from ParetoQ
![](images/349a0dde51edeabf24dc07fe2b67a52b208ee0c35d77dd0605c725db98697e9a.png)  


## Findings on QAT from ParetoQ
- Fine-tuning Characteristics

|  While fine-tuning enhances performance across all bit-widths, even binary and ternary, optimal fine-tuning effort inversely correlates with bit-width. For 3-bit and 4-bit weights, fine-tuning adjusts within a nearby grid to mitigate accuracy loss, and requires less finetuning tokens. In contrast, binary and ternary weights break the grid, creating new semantic representations to maintain performance, requiring longer finetuning. |
|:-|

<!-- 
从 fp 训练好的微调比 train from scratch 好，这和 1bit era 里的结论不同
高 bit qweight 和原始 weight 的l1 距离小
但是低bit 就相反
-->


## Findings on QAT from ParetoQ
![w:1100](images/58883ef809aff76aa02f8c8199a25c08309331e3d7e0b742eb320c37c4251bb9.png)  


## Findings on QAT from ParetoQ
![](images/5cffebb9cd2a51d768216cdb1db7561e1648e8689ddd9b26004679dd24da0080.png)  


## Findings on QAT from ParetoQ
- Quantization Method Choices

|Extreme low-bit quantization is highly sensitive to quantization function selection, with no single optimal function for all bit widths. Learnable range settings outperform statistics-based methods due to their flexibility in optimizing range parameters with respect to the final loss. Ternary and 2-bit quantization favor symmetric levels and balanced range coverage in quantization grid configuration, while imbalance levels with “0” in output levels are more effective for 3 and 4-bit quantization.|
|:-|

<!-- 
这个我们之前提到过了
不同 bitwidth 不同模型favor 的 quantizer 选择不一样
在 qat 的同时动态的学习量化参数是比较好的
低 bit 下喜欢关于 0 对称，高 bit下喜欢有0
-->



## Bonus: Extremely Low-bit LLMs
OneBit: **W1A16** quantized LLMs(!)
- 1-bit Linear Layer Architecture
$$
\begin{aligned}
    \mathbf{W}_{\pm 1} = \mathrm{Sign} \big ( \mathbf{W} \big ), \\
    \mathbf{Y} = \left [ \big ( \mathbf{X} \odot \mathbf{g} \big ) \mathbf{W}_{\pm 1}^{\mathrm{T}} \right ] \odot \mathbf{h}, \\
    \mathbf{Z} = \mathrm{LayerNorm} \big(\mathbf{Y}\big).
\end{aligned}
$$
where $\mathbf{g}$ and $\mathbf{h}$ are the two FP16 value vectors. (So in fact 1.0073 bit for a weight matrix with the shape $4096\times4096$.)
<!-- footer: Xu et al., [[arxiv]](https://arxiv.org/abs/2402.11295) -->

<!-- 
1bit 的线性层需要特殊设计，直接变成 1bit 的话显然没有价值了，因为相当于对于每个 block设置了一个阈值，超过这个阈值就是 1，否则就是-1
这样的量化精度显然是非常差的
他这里对于每个 weight，除了 1bit 的符号矩阵之外，还额外引入了两个全精度的向量
这个设计是非常有道理的，我们现在来看一下
-->


## Bonus: Extremely Low-bit LLMs
- Design intuition: Sign-Value-Independent Decomposition(SVID)
$$
\mathbf{W} = \mathbf{W}_{\pm 1} \odot \mathbf{W}_{\rm value}
$$
where $\mathbf{W}_{\rm value} = |\mathbf{W}|$.
- For $\mathbf{W}_{\rm value}$, we further approximately decompose it into the outer product of two vectors $a$ and $b$ (rank-1 approximation):
$$
\mathbf{W} \approx \mathbf{W}_{\mathrm{sign}} \odot \left ( \mathbf{a}\mathbf{b}^\mathrm{T} \right )
$$

<!-- 
这里我们发现，通过SVD 或者其他的方法，我们可以把 weight 近似为一个对应的符号矩阵和两个向量
-->


## Bonus: Extremely Low-bit LLMs
<style scoped>
p {
   font-size: 0.9rem;
}
li {
   font-size: 0.9rem;
}
</style>

Given the weight matrix $\mathbf{W}$ and input $\mathbf{X}$, the `Linear` layer can be reformulated as the following according to SVID:
$$
\mathbf{XW}^\mathrm{T} \approx \mathbf{X} \left( \mathbf{W}_{\mathrm{sign}} \odot \left ( \mathbf{a}\mathbf{b}^\mathrm{T} \right) \right) = 
\big[ \left( \mathbf{X} \odot \mathbf{b}^\mathrm{T} \right) \mathbf{W}_{\mathrm{sign}}^\mathrm{T} \big] \odot \mathbf{a}^\mathrm{T}.
$$

- This **bridges the gap between the architecture of the quantized model and its original weights**: the quantized model is an approximate initialization of the original model. 
- Compared to restoring the original matrix $\mathbf{W}$ first, **the purposed computational order saves memory** as there is no need to restore $\mathbf{W}$ in FP16 format.

<!-- 
这里的第二个等号是容易证明的
所以我们可以发现这里设计的 1bit linear 层其实就是对weight 做了一个 rank=1 的近似
这样交换顺序是为了减少计算过程中的显存开销
-->


## Bonus: Extremely Low-bit LLMs
Why not decompose $\mathbf{W}$ as $\mathbf{W} \approx \mathbf{a}\mathbf{b}^\mathrm{T}$ ?

**Proposition** Given matrices $\mathbf{W}$ and $| \mathbf{W} |$, $\mathbf{W} = \mathbf{W}_{\mathrm{sign}} \odot | \mathbf{W} |$. We decompose these matrices in the way $\mathbf{W} = \mathbf{ab}^\mathrm{T} + \mathbf{E}_1$ and $| \mathbf{W} | = \tilde{\mathbf{a}}\tilde{\mathbf{b}}^\mathrm{T} + \mathbf{E}_2$, where $\mathbf{E}_i$ denotes the error matrices. In terms of the Frobenius-norm, the SVID is closer to the original matrix $\mathbf{W}$:
$$
\left \| \mathbf{W} - \mathbf{W}_{\mathrm{sign}} \odot \tilde{\mathbf{a}}\tilde{\mathbf{b}}^\mathrm{T} \right \|_\mathrm{F}^2\le \left \| \mathbf{W} - \mathbf{ab}^\mathrm{T} \right \|_\mathrm{F}^2.
$$



## Bonus: Extremely Low-bit LLMs
- SVID provides an effective starting point for further training.
- Employ quantization-aware knowledge distillation to transfer knowledge from the original model.
$$
\begin{gather}
\mathcal{L}_{\mathrm{CE}}=-\frac{1}{n_s}\sum_{i=1}^{n_s}\sum_{c}P_c^{\mathcal{T}}\left (\mathbf{o}_i \right )\mathrm{log}P_c^{\mathcal{S}}\left (\mathbf{o}_i \right ), \\
\mathcal{L}_{\mathrm{MSE}}=\sum_{i=1}^{n_s}\sum_{j=1}^{n_l}\Bigg\| \frac{\mathbf{q}_{i,j}^{\mathcal{T}}}{\left \| \mathbf{q}_{i,j}^{\mathcal{T}} \right\|_2} - \frac{\mathbf{q}_{i,j}^{\mathcal{S}}}{\left \| \mathbf{q}_{i,j}^{\mathcal{S}} \right\|_2} \Bigg\|_2^2.
\end{gather}
$$

<!-- 
SVID不是为了完全恢复 weight，而是作为 qat 训练的起点
利用 LLM-QAT 中的蒸馏的 loss 和类似EfficientQAT 中的 reconstruction的 loss，可以对 1bit 模型进一步训练
-->


## Bonus: Extremely Low-bit <span class="morph" style="--morph-name:llms;">LLMs</span>
![w:650](images/900ddf0e26e514c369e8a49de087d7752ccf06c5d873cf63e4e6bce619e2bd00.png)  

<!-- 
注意这里除了 Onebit 其他的方法都用的是W2A16
-->


# Evaluating Quantized <span class="morph" style="--morph-name:llms;">LLMs</span>
<!-- _class: lead -->
<!-- footer: Li et al., [[arxiv]](https://arxiv.org/abs/2402.18158) -->

## Evaluating Quantized LLMs
![bg w:800](images/f1d86fbd6860621bb617d4092ec7e3c0823234618ca4e57f3ecc0b9d9905ab7b.png)  

<!-- 
估计讲到这里已经超时了
大概放了一张key findings
有兴趣可以看这个论文，个人感觉实验做的非常solid
-->


# <!-- fit -->Thanks for Watching
Pingzhi Tang
stanleytang@stu.pku.edu.cn
<!-- class: lead -->
<!-- footer: \b -->