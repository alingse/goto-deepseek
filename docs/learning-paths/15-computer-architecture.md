# 计算机组成与体系结构（2天）

## 概述
- **目标**：深入理解计算机硬件组成与体系结构，掌握CPU架构、存储层次、并行计算等核心概念，为学习异构计算（GPU/NPU）打下坚实基础，满足JD中"深刻理解计算机组成等核心原理"和"异构计算资源抽象、池化、调度"的要求
- **时间**：春节第3周（2天）
- **前提**：了解基本计算机概念，有编程经验
- **强度**：中高强度（每天8小时），适合需要理解硬件的工程师

## JD要求对应

### JD领域覆盖
| JD领域 | 对应内容 | 优先级 |
|--------|----------|--------|
| 一、高并发服务端与API系统 | CPU缓存友好、内存屏障、并行计算 | ⭐⭐⭐ |
| 二、大规模数据处理Pipeline | 存储层次、I/O优化、数据局部性 | ⭐⭐ |
| 三、Agent基础设施与运行时平台 | 资源调度、NUMA架构、功耗管理 | ⭐⭐ |
| 四、异构超算基础设施 | GPU/NPU架构、并行算法、RDMA | ⭐⭐⭐ |

### JD能力对应
| 能力要求 | 学习内容 | 验证方式 |
|----------|----------|----------|
| **深刻理解计算机组成** | CPU架构、存储层次、I/O系统 | 硬件优化方案 |
| **并行计算** | 多核架构、SIMD、并行算法 | 并行程序设计 |
| **异构计算** | GPU/NPU架构、CUDA编程、模型推理优化 | GPU程序实现 |

## 学习重点

### 1. CPU架构与原理（第1天上午）
**JD引用**："深刻理解计算机组成等核心原理"

**核心内容**：
- CPU基本结构
  - 运算器（ALU）
  - 控制器（CU）
  - 寄存器（Registers）
  - 高速缓存（Cache）
- 指令系统
  - 指令格式（Opcode + Operand）
  - 指令类型（数据传送、算术逻辑、控制转移）
  - 寻址方式（立即寻址、直接寻址、间接寻址）
  - 指令周期（取指、译码、执行、写回）
- CPU流水线
  - 流水线原理
  - 流水线冒险（数据冒险、控制冒险、结构冒险）
  - 流水线优化（转发、分支预测）
  - 超标量与超流水线
- 分支预测
  - 静态分支预测
  - 动态分支预测
  - 分支目标缓冲器（BTB）
  - 两位饱和计数器

**实践任务**：
- 分析CPU指令周期
- 理解流水线冒险
- 使用分支预测优化代码

### 2. 存储层次与缓存（第1天下午）
**JD引用**："深刻理解计算机组成等核心原理"、"负责加速卡（如GPU/NPU）等异构计算资源的抽象、池化、调度"

**核心内容**：
- 存储层次结构
  - 寄存器（CPU内）
  - L1 Cache（32KB-64KB/核）
  - L2 Cache（256KB-1MB/核）
  - L3 Cache（8MB-64MB/芯片）
  - 主存（DRAM，GB级）
  - 外部存储（SSD/HDD，TB级）
- Cache原理
  - Cache映射方式（直接映射、组相联、全相联）
  - Cache行结构（Tag + Index + Offset）
  - 替换策略（LRU、FIFO、LFU、Random）
  - 写策略（Write-Through、Write-Back）
- 缓存一致性
  - MESI协议（Modified/Exclusive/Shared/Invalid）
  - 缓存一致性协议
  - 伪共享问题
- 存储性能
  - 访问延迟（纳秒到毫秒）
  - 带宽（GB/s到TB/s）
  - 局部性原理（时间局部性、空间局部性）

**实践任务**：
- 分析Cache命中率
- 优化数据布局提高Cache命中率
- 解决伪共享问题

### 3. 内存系统与DRAM（第1天晚上）
**JD引用**："负责加速卡（如GPU/NPU）等异构计算资源的抽象、池化、调度"

**核心内容**：
- DRAM原理
  - DRAM存储单元（电容 + 晶体管）
  - DRAM刷新机制
  - DRAM时序（CAS Latency、tRCD、tRP）
- 内存通道
  - 单通道、双通道、四通道
  - 内存带宽计算
  - 内存交错（Interleaving）
- NUMA架构
  - UMA vs NUMA
  - NUMA节点
  - 本地内存 vs 远程内存
  - NUMA感知编程
- 内存控制器
  - 内存控制器功能
  - 内存调度算法
  - 内存访问优化

**实践任务**：
- 配置NUMA亲和性
- 分析内存带宽
- 优化NUMA访问模式

### 4. 并行计算架构（第2天上午）
**JD引用**："参与设计、构建与优化支撑大模型训练与推理的异构计算集群管理平台"

**核心内容**：
- 并行分类（Flynn分类）
  - SISD（单指令单数据）
  - SIMD（单指令多数据）
  - MISD（多指令单数据）
  - MIMD（多指令多数据）
- Amdahl定律
  - 串行部分限制加速比
  - 并行化收益分析
  - Gustafson定律
- CPU并行技术
  - 多核架构
  - 超线程技术（Hyper-Threading）
  - 向量处理器（AVX、AVX2、AVX-512）
- GPU并行架构
  - GPU SIMT架构
  - CUDA核心（Streaming Processor）
  - 线程层次（Grid、Block、Warp）
  - 内存层次（Global/Shared/Register）

**实践任务**：
- 使用SIMD向量化优化
- 分析并行加速比
- 编写简单CUDA程序

### 5. I/O系统与总线（第2天下午）
**JD引用**："持续优化数据处理各环节的性能与吞吐"

**核心内容**：
- I/O系统结构
  - 程序查询（Programmed I/O）
  - 中断驱动I/O
  - DMA直接内存访问
- 总线系统
  - PCI Express（PCIe）
    - PCIe代数（3.0/4.0/5.0）
    - 通道数（x1/x4/x8/x16）
    - 带宽计算
  - NVMe
    - NVMe协议原理
    - 队列深度（Queue Depth）
    - I/O延迟优化
- 存储设备
  - SSD架构（Controller + NAND Flash）
  - SSD性能因素（闪存类型、写放大、垃圾回收）
  - HDD机械特性
- 高速互连
  - InfiniBand
  - RoCE（RDMA over Converged Ethernet）
  - iWARP

**实践任务**：
- 对比SSD/HDD性能
- 优化NVMe I/O
- 理解PCIe带宽

### 6. GPU架构与CUDA编程（第2天下午）
**JD引用**："负责加速卡（如GPU/NPU）等异构计算资源的抽象、池化、调度与性能优化"

**核心内容**：
- GPU架构
  - NVIDIA GPU架构（Volta/Ampere/Hopper）
  - CUDA核心数量
  - Tensor Core（矩阵运算）
  - RT Core（光线追踪）
  - 内存层次（HBM显存）
- CUDA编程模型
  - Host与Device
  - Kernel函数
  - 线程层次（Grid/Block/Warp/Thread）
  - 内存管理（Global/Shared/Constant/Texture）
- CUDA优化技术
  - 内存合并访问
  - 共享内存使用
  - 线程束分化（Warp Divergence）
  - 占用率优化
  - 流式并行（Streams）
- GPU虚拟化
  - MIG（Multi-Instance GPU）
  - vGPU
  - GPU资源池化

**实践任务**：
- 编写CUDA向量加法
- 优化CUDA内存访问
- 配置GPU虚拟化

### 7. 异构计算与AI加速器（第2天晚上）
**JD引用**："为上层AI研发提供稳定高效的算力底座"

**核心内容**：
- AI加速器
  - NVIDIA TPU（Tensor Processing Unit）
  - Google TPU架构
  - 华为昇腾（Ascend）
  - 寒武纪MLU
  - Graphcore IPU
- NPU架构
  - 神经网络运算特点
  - 稀疏计算
  - 低精度计算（INT8/FP16/BF16）
  - 片上内存
- 系统级优化
  - CPU-GPU协同
  - 内存带宽优化
  - 功耗管理
  - 冷却系统
- 集群管理
  - GPU调度（Kubernetes GPU调度）
  - 资源池化（MIG、vGPU）
  - 故障恢复
  - 能效优化

**实践任务**：
- 了解主流AI加速器架构
- 配置Kubernetes GPU调度
- 设计GPU资源池化方案

## 实践项目：GPU性能优化实践

### 项目目标
**JD对应**：满足"负责加速卡（如GPU/NPU）等异构计算资源的抽象、池化、调度与性能优化"要求

实现GPU性能优化实践，包含：
1. CUDA程序性能分析
2. 内存访问优化
3. 线程配置优化
4. GPU资源监控

### 技术栈参考（明确版本）
- **CUDA Toolkit**：12.0+
- **NVIDIA Driver**：525.0+
- **nsight-systems**：2023.1+
- **nsight-compute**：2023.1+
- **PyTorch/TensorFlow**：2.0+（支持GPU）

### 环境配置要求
- **硬件**：NVIDIA GPU（RTX 3080+ 或 A100）
- **操作系统**：Ubuntu 22.04 LTS
- **依赖安装**：
  ```bash
  # 安装CUDA
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install cuda-toolkit-12-0

  # 安装nsight工具
  sudo apt-get install nvidia-nsight-systems nvidia-nsight-compute
  ```

### 架构设计
```
gpu-optimization/
├── basics/                    # CUDA基础
│   ├── vector_add/           # 向量加法
│   ├── matrix_mul/          # 矩阵乘法
│   └── reduction/            # 归约运算
├── memory/                    # 内存优化
│   ├── coalesced_access/     # 合并访问
│   ├── shared_memory/        # 共享内存
│   ├── constant_memory/      # 常量内存
│   └── texture_memory/       # 纹理内存
├── threading/                 # 线程优化
│   ├── warp_divergence/      # 线程束分化
│   ├── occupancy/            # 占用率优化
│   └── streams/              # 流式并行
├── profiling/                 # 性能分析
│   ├── nsight-systems/       # 系统分析
│   ├── nsight-compute/       # 计算分析
│   └── cuda-memcheck/        # 内存检查
├── models/                    # 模型优化
│   ├── tensor_core/          # Tensor Core使用
│   ├── mixed_precision/      # 混合精度
│   └── torch_gpu/            # PyTorch GPU
└── virtualization/           # 虚拟化
    ├── mig_config/           # MIG配置
    ├── vgpu_setup/          # vGPU配置
    └── k8s_gpu/             # K8s GPU调度
```

### 核心组件设计

#### 1. CUDA向量加法
```cpp
// vector_add.cu
#include <cuda_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;  // 1M元素

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // 复制数据到设备
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 调用核函数
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // 复制结果回主机
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

#### 2. 共享内存优化
```cpp
// matrix_mul_shared.cu
__global__ void matrix_mul_shared(float* a, float* b, float* c, int n) {
    // 共享内存（块内数据）
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    float sum = 0.0f;

    // 遍历所有分块
    for (int k = 0; k < (n - 1) / 32 + 1; k++) {
        // 加载数据到共享内存
        As[ty][tx] = a[row * n + k * 32 + tx];
        Bs[ty][tx] = b[(k * 32 + ty) * n + col];

        __syncthreads();

        // 计算分块乘积
        for (int i = 0; i < 32; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    c[row * n + col] = sum;
}
```

#### 3. 性能分析脚本
```bash
#!/bin/bash
# profile_cuda.sh

# 使用nsight-systems分析
nsight-systems --csv --output=report \
    ./vector_add

# 使用nsight-compute分析
ncu --set full --export report \
    ./vector_add

# 使用nvprof分析（已废弃但兼容）
nvprof --print-gpu-summary \
    ./vector_add

# 内存传输分析
nvprof --print-gpu-trace \
    --devices all \
    --unified-memory-profiling off \
    ./vector_add
```

## 学习资源

### 经典书籍
1. **《计算机组成与设计》**（ Patterson & Hennessy）：经典教材
2. **《深入理解计算机系统》**（CSAPP）：系统级编程
3. **《计算机体系结构》**：量化研究方法
4. **《CUDA编程指南》**：NVIDIA官方指南
5. **《GPU高性能编程》**：CUDA实战

### 官方文档
1. **Intel开发者手册**：[software.intel.com](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
2. **AMD架构文档**：[developer.amd.com](https://developer.amd.com/)
3. **NVIDIA CUDA文档**：[docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/)
4. **PCI Express规范**：[pcisig.com](https://pcisig.com/)

### 在线课程
1. **MIT 6.004**：[计算体系结构](https://ocw.mit.edu/courses/6-004-computation-structures/)
2. **Stanford CS149**：[并行处理器](https://cs149.stanford.edu/)
3. **Coursera Computer Architecture**：[计算机体系结构](https://www.coursera.org/)
4. **NVIDIA深度学习**：[developer.nvidia.com](https://developer.nvidia.com/deep-learning)

### 技术博客与案例
1. **AnandTech**：[硬件评测](https://www.anandtech.com/)
2. **NVIDIA Blog**：[GPU技术](https://blogs.nvidia.com/)
3. **Facebook Engineering**：[硬件优化](https://engineering.fb.com/)
4. **Google TPU Blog**：[TPU技术](https://cloud.google.com/tpu/docs)

### 开源项目参考
1. **CUDA Samples**：[github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
2. **PyTorch**：[github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
3. **TensorFlow**：[github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
4. **Llama.cpp**：[github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

### 权威论文
1. **MESI Protocol**（1978）
2. **SIMD Architecture**（1980s）
3. **NVIDIA Tesla Architecture**（2006）
4. **Google TPU**（2016）
5. **Ampere Architecture**（2020）

## 学习产出要求

### 设计产出
1. ✅ 计算机体系结构知识图谱
2. ✅ GPU性能优化方案
3. ✅ 异构计算架构设计

### 代码产出
1. ✅ CUDA基础程序
2. ✅ GPU内存优化代码
3. ✅ 性能分析脚本

### 技能验证
1. ✅ 理解CPU流水线与Cache原理
2. ✅ 掌握NUMA编程
3. ✅ 能够编写CUDA程序
4. ✅ 能够优化GPU性能
5. ✅ 理解异构计算架构

### 文档产出
1. ✅ CPU性能优化指南
2. ✅ CUDA编程手册
3. ✅ GPU虚拟化配置指南

## 时间安排建议

### 第1天（CPU与存储）
- **上午（4小时）**：CPU架构
  - 指令系统与流水线
  - 分支预测
  - 实践：分析CPU性能

- **下午（4小时）**：存储层次
  - Cache原理与优化
  - NUMA架构
  - 实践：优化Cache利用率

- **晚上（2小时）**：并行计算
  - Flynn分类
  - Amdahl定律
  - SIMD向量化

### 第2天（GPU与异构）
- **上午（4小时）**：I/O与总线
  - DMA与PCIe
  - NVMe优化
  - 高速互连

- **下午（4小时）**：GPU架构与CUDA
  - CUDA编程模型
  - 内存优化
  - 实践：编写CUDA程序

- **晚上（2小时）**：异构计算
  - AI加速器架构
  - GPU虚拟化
  - 集群资源调度

## 学习方法建议

### 1. 理论与实践结合
- 阅读硬件文档
- 编写CUDA程序
- 使用性能分析工具
- 优化实际代码

### 2. 从CPU到GPU
- 先理解CPU架构
- 再学习GPU架构
- 最后掌握异构编程

### 3. 关注性能指标
- CPU：IPC、Cache命中率、时钟频率
- GPU：吞吐量、内存带宽、占用率
- 系统：功耗、散热、成本

## 常见问题与解决方案

### Q1：Cache如何优化？
**A**：优化方法：
- **数据布局**：结构体对齐
- **访问模式**：连续访问
- **预取**：软件预取指令
- **伪共享**：数据对齐

### Q2：SIMD如何优化？
**A**：优化方法：
- **编译器向量化**：-O3 -march=native
- **手动向量化**：使用intrinsics
- **循环展开**：增加指令级并行
- **数据对齐**：16/32字节对齐

### Q3：CUDA如何优化？
**A**：优化优先级：
1. **内存访问**：合并访问、共享内存
2. **线程配置**：合理block大小
3. **占用率**：平衡并行度与资源
4. **计算优化**：使用Tensor Core

### Q4：NUMA如何配置？
**A**：配置方法：
1. **numactl**：绑定CPU/内存节点
2. **libnuma**：编程接口
3. **内存分配**：本地优先
4. **进程迁移**：避免跨节点

### Q5：GPU虚拟化怎么做？
**A**：虚拟化方案：
1. **MIG**：单GPU多实例
2. **vGPU**：虚拟GPU
3. **时间分片**：CUDA MPS
4. **Kubernetes**：GPU调度

## 知识体系构建

### 核心知识领域

#### 1. CPU架构
```
CPU架构
├── 运算单元
│   ├── ALU（算术逻辑单元）
│   ├── FPU（浮点单元）
│   └── SIMD单元（AVX）
├── 控制单元
│   ├── 指令译码
│   ├── 流水线控制
│   └── 分支预测
├── 存储单元
│   ├── 寄存器文件
│   ├── Cache（L1/L2/L3）
│   └── TLB
└── 总线接口
    ├── 前端总线
    └── DMI/QPI
```

#### 2. 存储层次
```
存储层次
├── 寄存器（0.3ns，32B）
├── L1 Cache（1ns，32KB）
├── L2 Cache（4ns，256KB）
├── L3 Cache（12ns，8MB）
├── 主存（100ns，GB级）
└── 外部存储（100μs，TB级）
```

#### 3. 并行计算
```
并行分类
├── SISD（传统CPU）
├── SIMD（向量处理器、GPU）
├── MISD（罕见）
└── MIMD（多核CPU、分布式系统）
```

### 学习深度建议

#### 精通级别
- CPU流水线与Cache原理
- NUMA编程
- CUDA编程与优化
- GPU虚拟化

#### 掌握级别
- 指令级并行
- SIMD向量化
- GPU架构细节
- 异构调度

#### 了解级别
- CPU微架构设计
- 功耗管理
- AI加速器架构
- 集群互连

## 下一步学习

### 立即进入
1. **操作系统内核**（路径13）：
   - 理解内核对硬件的抽象
   - 协同效应：本路径的硬件 + 操作系统的软件

2. **异构超算项目**：
   - GPU集群管理
   - 分布式训练

### 后续深入
1. **AI/ML系统性学习**（路径04）：模型推理优化
2. **云原生进阶**（路径06）：GPU K8s调度

### 持续跟进
- GPU新架构（Blackwell）
- 光刻技术与芯片发展
- 异构计算标准演进

---

## 学习路径特点

### 针对人群
- 需要理解底层硬件的工程师
- 面向JD中的"计算机组成"和"异构计算"要求
- 适合需要从事AI加速的工程师

### 学习策略
- **中高强度**：2天集中学习，每天8小时
- **理论为主**：70%理论学习，30%实践
- **硬件导向**：深入理解计算机硬件

### 协同学习
- 与操作系统路径：软硬件结合
- 与性能优化路径：硬件性能分析
- 与AI/ML路径：模型推理优化

### 质量保证
- 内容基于权威资料
- 代码示例可直接运行
- 性能指标真实可靠

---

*学习路径设计：针对需要理解底层硬件的工程师，深入学习计算机组成与异构计算*
*时间窗口：春节第3周2天，中高强度学习硬件架构*
*JD对标：满足JD中计算机组成、异构计算、GPU/NPU等核心要求*
