# 操作系统内核与原理（3天）

## 概述
- **目标**：深入理解操作系统内核原理，掌握进程管理、内存管理、文件系统、进程间通信等核心机制，满足JD中"深刻理解操作系统等核心原理"和"构建高性能、高安全性的Agent运行时环境"的要求
- **时间**：春节第2周后半段（3天）
- **前提**：熟悉C语言，了解基本操作系统概念
- **强度**：高强度（每天8-10小时），适合需要深入理解系统的工程师

## JD要求对应

### JD领域覆盖
| JD领域 | 对应内容 | 优先级 |
|--------|----------|--------|
| 一、高并发服务端与API系统 | 多进程/多线程并发、进程间通信、性能调优 | ⭐⭐⭐ |
| 二、大规模数据处理Pipeline | 内存管理、文件系统、I/O性能优化 | ⭐⭐⭐ |
| 三、Agent基础设施与运行时平台 | 容器原理（Namespace、Cgroup）、资源隔离、安全机制 | ⭐⭐⭐ |
| 四、异构超算基础设施 | 进程调度、NUMA架构、内核性能优化 | ⭐⭐ |

### JD能力对应
| 能力要求 | 学习内容 | 验证方式 |
|----------|----------|----------|
| **深刻理解操作系统** | 进程、线程、内存、文件系统、内核机制 | 内核模块开发 |
| **容器与虚拟化** | Namespace、Cgroup、容器运行时 | 容器实现 |
| **性能优化能力** | 系统调用、内核调优、Profiling工具 | 性能分析与调优 |
| **系统安全** | 权限管理、安全机制、内核安全 | 安全配置 |

## 学习重点

### 1. 进程与线程（第1天上午）
**JD引用**："深刻理解操作系统等核心原理"

**核心内容**：
- 进程概念
  - 进程定义与特征
  - 进程控制块（PCB）
  - 进程状态转换（新建、就绪、运行、阻塞、终止）
  - 进程上下文切换
- 线程概念
  - 线程 vs 进程对比
  - 线程模型（1:1、N:1、混合）
  - 用户态线程 vs 内核态线程
  - 协程与纤程
- 进程调度算法
  - FCFS（先来先服务）
  - SJF（短作业优先）
  - 时间片轮转
  - 优先级调度
  - 多级反馈队列
  - Linux CFS（完全公平调度器）
- 进程控制
  - fork()创建进程
  - exec()加载程序
  - wait()/waitpid()等待子进程
  - exit()终止进程
  - 僵尸进程与孤儿进程

**实践任务**：
- 使用fork()创建父子进程
- 实现简单的进程调度器模拟
- 分析进程上下文切换开销
- 使用top/ps观察进程状态

### 2. 线程与并发（第1天下午）
**JD引用**："构建高性能、高安全性的Agent运行时环境"

**核心内容**：
- 线程API
  - pthread_create()创建线程
  - pthread_join()等待线程
  - pthread_exit()退出线程
  - 线程属性设置
- 线程同步
  - 互斥锁（Mutex）
  - 读写锁（ReadWriteLock）
  - 条件变量（Condition Variable）
  - 信号量（Semaphore）
  - 自旋锁（Spinlock）
  - RCU（Read-Copy-Update）
- 线程安全
  - 竞态条件（Race Condition）
  - 死锁（Deadlock）
  - 活锁（Livelock）
  - 饥饿（Starvation）
- 无锁编程
  - CAS（Compare-And-Swap）
  - 原子操作（Atomic Operations）
  - 内存屏障（Memory Barrier）
  - Seqlock与Seqlock

**实践任务**：
- 使用pthread实现多线程程序
- 实现生产者-消费者模型
- 实现读写锁
- 使用CAS实现无锁队列

### 3. 内存管理（第1天晚上）
**JD引用**："深刻理解计算机组成、操作系统、计算机网络等核心原理"

**核心内容**：
- 内存层次结构
  - 寄存器
  - Cache（L1/L2/L3）
  - 主存（DRAM）
  - 虚拟内存
- 虚拟内存
  - 虚拟地址 vs 物理地址
  - 页表（Page Table）
  - 多级页表
  - TLB（Translation Lookaside Buffer）
  - 缺页中断（Page Fault）
- 内存分配
  - 内存分区（固定分区、可变分区）
  - 分页管理（Paging）
  - 分段管理（Segmentation）
  - 段页式管理
- 内存分配器
  - malloc/free实现原理
  - jemalloc
  - tcmalloc
  - 内存碎片（内碎片、外碎片）
- 内存优化
  - 大页内存（Huge Pages）
  - 内存池（Memory Pool）
  - 对象池（Object Pool）
  - slab分配器

**实践任务**：
- 实现简单的内存分配器
- 分析malloc内存布局
- 配置Huge Pages
- 使用valgrind检测内存泄漏

### 4. 文件系统（第2天上午）
**JD引用**："负责数据采集、清洗、去重与质量评估系统的设计与开发"

**核心内容**：
- 文件系统基础
  - 文件概念与属性
  - 文件类型（普通文件、目录、设备文件、链接等）
  - 文件操作（open、read、write、close、lseek）
  - 文件描述符
- 文件系统类型
  - ext4（第四代扩展文件系统）
  - xfs（高性能日志文件系统）
  - btrfs（写时复制文件系统）
  - zfs（企业级文件系统）
- VFS（虚拟文件系统）
  - VFS架构
  - super_block、inode、dentry、file
  - 文件系统注册与挂载
- 日志文件系统
  - 日志原理
  - 日志恢复
  - 写时复制（Copy-on-Write）
- 分布式文件系统
  - NFS（网络文件系统）
  - Ceph（统一分布式存储）
  - HDFS（Hadoop分布式文件系统）

**实践任务**：
- 实现简单的文件系统
- 使用strace追踪文件系统调用
- 对比ext4/xfs性能
- 配置NFS共享

### 5. 进程间通信（第2天下午）
**JD引用**："能够设计高可用、高可靠的系统架构"

**核心内容**：
- IPC基础概念
  - 为什么需要IPC
  - IPC分类（数据传输、共享内存、远程调用）
- 管道（Pipe）
  - 匿名管道
  - 命名管道（FIFO）
  - 管道特点与限制
- 消息队列（Message Queue）
  - System V消息队列
  - POSIX消息队列
  - 消息队列特点
- 共享内存（Shared Memory）
  - System V共享内存
  - POSIX共享内存
  - 同步机制配合
- 信号量（Semaphore）
  - System V信号量
  - POSIX信号量
  - 信号量集
- 信号（Signal）
  - 信号类型与处理
  - signal()与sigaction()
  - 可靠信号与不可靠信号
- Unix Domain Socket
  - 本地进程间通信
  - Socket API
  - vs 网络Socket对比

**实践任务**：
- 实现管道通信
- 实现消息队列通信
- 实现共享内存同步
- 实现Unix Domain Socket通信

### 6. Linux内核机制（第2天晚上）
**JD引用**："构建高性能、高安全性的Agent运行时环境"

**核心内容**：
- 内核架构
  - 宏内核 vs 微内核
  - Linux内核架构
  - 内核态 vs 用户态
  - 系统调用机制
- 内核模块
  - 模块原理
  - 模块编程
  - /proc文件系统
  - sysfs文件系统
- 内核同步
  - 自旋锁
  - 互斥锁
  - 读写锁
  - 原子操作
  - RCU（Read-Copy-Update）
- 内核调试
  - printk()日志
  - /proc/meminfo
  - ftrace追踪
  - perf性能分析
  - crash工具

**实践任务**：
- 编写简单的内核模块
- 创建/proc文件
- 使用ftrace追踪内核函数
- 分析内核崩溃日志

### 7. 容器原理（第3天上午）
**JD引用**："设计与开发支撑海量AI Agent运行的下一代容器调度与隔离平台"

**核心内容**：
- Namespace（命名空间）
  - UTS（主机名与域名）
  - IPC（进程间通信隔离）
  - PID（进程ID隔离）
  - Network（网络隔离）
  - Mount（挂载点隔离）
  - User（用户与用户组隔离）
  - Cgroup（命名空间）
- Cgroup（控制组）
  - Cgroup原理
  - Cgroup子系统（cpu、memory、blkio、net_cls）
  - 资源限制（CPU、内存、磁盘I/O）
  - 资源监控
- 容器运行时
  - runc（OCI运行时）
  - containerd（容器管理）
  - CRI-O（Kubernetes运行时）
  - gVisor（用户态内核）
  - Kata Containers（轻量级虚拟机）
- 容器网络
  - veth pair
  - bridge网络
  - overlay网络
  - CNI（容器网络接口）

**实践任务**：
- 使用unshare创建命名空间
- 使用cgroup限制资源
- 编写简单的容器运行时
- 配置容器网络

### 8. eBPF与内核追踪（第3天下午）
**JD引用**："熟练运用Profiling和可观测性工具分析与定位复杂系统问题"

**核心内容**：
- eBPF基础
  - eBPF原理
  - BPF程序类型
  - BPF虚拟机
  - BPF maps
- eBPF编程
  - BCC（BPF Compiler Collection）
  - libbpf
  - CO-RE（Compile Once, Run Everywhere）
  - bpftrace
- eBPF应用场景
  - 网络监控与过滤（XDP）
  - 性能分析（函数追踪、延迟分析）
  - 安全监控（系统调用监控）
  - 可观测性（监控指标、日志）
- eBPF工具
  - bpftrace（动态追踪）
  - bcc工具集
  - libbpf-bootstrap
  - BPF Performance Tools

**实践任务**：
- 编写eBPF程序监控系统调用
- 使用bpftrace追踪函数
- 使用bcc工具分析性能
- 实现XDP网络过滤

## 实践项目：实现简易容器运行时

### 项目目标
**JD对应**：满足"设计与开发支撑海量AI Agent运行的下一代容器调度与隔离平台"要求

实现一个简易的容器运行时，支持：
1. Namespace隔离（UTS、PID、Mount、Network）
2. Cgroup资源限制（CPU、内存）
3. 文件系统隔离（chroot、pivot_root）
4. 进程监控与管理

### 技术栈参考（明确版本）
- **编程语言**：C（C11）或 Go 1.21+
- **系统调用**：Linux系统调用（unshare、clone、pivot_root）
- **依赖**：libseccomp 2.5+、glibc 2.35+
- **工具**：strace、perf、eBPF

### 环境配置要求
- **操作系统**：Linux 5.15+（推荐Ubuntu 22.04）
- **内核版本**：5.15+（支持所有Namespace特性）
- **编译工具**：GCC 11+ / Clang 14+
- **依赖安装**：
  ```bash
  # 安装依赖
  sudo apt-get install libseccomp-dev libcap-dev

  # 安装工具
  sudo apt-get install strace perf bpfcc-tools linux-headers-$(uname -r)

  # 验证内核支持
  ls /proc/self/ns/
  ```

### 架构设计
```
simple-container/
├── cmd/                   # 命令行工具
│   ├── run.c/.go                # 容器运行命令
│   ├── exec.c/.go               # 容器内执行命令
│   ├── ps.c/.go                 # 列出容器进程
│   └── stop.c/.go               # 停止容器
├── runtime/               # 运行时核心
│   ├── container.h/.go          # 容器结构
│   ├── namespace.h/.go          # Namespace隔离
│   ├── cgroup.h/.go             # Cgroup限制
│   ├── mount.h/.go              # 文件系统挂载
│   └── network.h/.go            # 网络配置
├── isolation/             # 隔离机制
│   ├── uts_ns.c/.go             # UTS命名空间
│   ├── pid_ns.c/.go             # PID命名空间
│   ├── mount_ns.c/.go           # Mount命名空间
│   ├── net_ns.c/.go             # Network命名空间
│   └── user_ns.c/.go            # User命名空间
├── resources/             # 资源管理
│   ├── cpu.c/.go                # CPU限制
│   ├── memory.c/.go             # 内存限制
│   └── monitor.c/.go            # 资源监控
├── security/               # 安全机制
│   ├── seccomp.c/.go            # Seccomp过滤
│   ├── capabilities.c/.go       # 能力管理
│   └── AppArmor.c/.go           # AppArmor配置
├── storage/                # 存储管理
│   ├── rootfs.c/.go             # 根文件系统
│   ├── overlay.c/.go            # OverlayFS
│   └── volume.c/.go             # 卷管理
├── network/                # 网络管理
│   ├── bridge.c/.go             # 网桥配置
│   ├── veth.c/.go               # veth pair
│   ├── iptables.c/.go           # 防火墙规则
│   └── dns.c/.go                # DNS配置
├── monitor/                # 监控追踪
│   ├── ebpf.c/.go               # eBPF监控
│   ├── perf.c/.go               # 性能分析
│   └── metrics.c/.go            # 指标收集
└── examples/               # 示例
    ├── alpine container         # 运行Alpine
    ├── nginx container          # 运行Nginx
    └── custom container         # 自定义容器
```

### 核心组件设计

#### 1. Namespace隔离
```c
// 创建子进程并隔离Namespace
int create_container(char **cmd) {
    int pid = clone(
        child_func,                    // 子进程函数
        stack + STACK_SIZE,            // 栈地址
        CLONE_NEWUTS       |           // UTS命名空间（主机名）
        CLONE_NEWPID       |           // PID命名空间（进程ID）
        CLONE_NEWNS        |           // Mount命名空间（挂载点）
        CLONE_NEWNET       |           // Network命名空间（网络）
        SIGCHLD,                       // 子进程退出信号
        NULL                           // 参数
    );

    if (pid < 0) {
        perror("clone");
        return -1;
    }

    return pid;
}

// 子进程函数
int child_func(void *arg) {
    // 设置主机名
    sethostname("container", 9);

    // 挂载proc文件系统
    mount("proc", "/proc", "proc", 0, NULL);

    // 执行用户命令
    execvp(cmd[0], cmd);

    return 0;
}
```

#### 2. Cgroup资源限制
```c
// 设置CPU限制
int set_cpu_limit(const char *cgroup_path, int cpu_quota) {
    char path[256];
    snprintf(path, sizeof(path), "%s/cpu.cfs_quota_us", cgroup_path);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    fprintf(fp, "%d", cpu_quota * 100000);  // 转换为微秒
    fclose(fp);

    return 0;
}

// 设置内存限制
int set_memory_limit(const char *cgroup_path, int memory_mb) {
    char path[256];
    snprintf(path, sizeof(path), "%s/memory.limit_in_bytes", cgroup_path);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    fprintf(fp, "%d", memory_mb * 1024 * 1024);  // 转换为字节
    fclose(fp);

    return 0;
}
```

#### 3. 文件系统隔离
```c
// pivot_root切换根文件系统
int setup_rootfs(const char *rootfs) {
    // 挂载新的根文件系统
    if (mount(rootfs, rootfs, "bind", MS_BIND | MS_REC, NULL) < 0) {
        perror("mount");
        return -1;
    }

    // 创建old root目录
    char pivot_dir[256];
    snprintf(pivot_dir, sizeof(pivot_dir), "%s/.pivot_root", rootfs);
    mkdir(pivot_dir, 0700);

    // 切换根文件系统
    if (pivot_root(rootfs, pivot_dir) < 0) {
        perror("pivot_root");
        return -1;
    }

    // 卸载old root
    chdir("/");
    umount2(".pivot_root", MNT_DETACH);
    rmdir(".pivot_root");

    return 0;
}
```

#### 4. 网络配置
```c
// 创建veth pair
int create_veth_pair(const char *veth1, const char *veth2) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
        "ip link add %s type veth peer name %s",
        veth1, veth2);

    return system(cmd);
}

// 将veth移到容器网络命名空间
int move_veth_to_container(const char *veth, int pid) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
        "ip link set %s netns %d",
        veth, pid);

    return system(cmd);
}

// 配置容器网络
int setup_container_network(const char *ip, const char *mask) {
    char cmd[256];

    // 配置IP地址
    snprintf(cmd, sizeof(cmd),
        "ip addr add %s/%s dev eth0",
        ip, mask);
    system(cmd);

    // 启动网卡
    system("ip link set eth0 up");

    // 配置路由
    snprintf(cmd, sizeof(cmd),
        "ip route add default via %s",
        GATEWAY_IP);
    system(cmd);

    return 0;
}
```

## 学习资源

### 经典书籍
1. **《现代操作系统》**（第4版）：Andrew S. Tanenbaum - OS经典教材
2. **《操作系统导论》（OSTEP）**：Remzi & Andrea - 交互式学习
3. **《深入理解Linux内核》**：Daniel P. Bovet - 内核原理
4. **《UNIX环境高级编程》（APUE）**：W. Richard Stevens - Unix系统编程
5. **《Linux内核设计与实现》**：Robert Love - 内核实现

### 官方文档
1. **Linux内核文档**：[The Linux Kernel documentation](https://www.kernel.org/doc/html/latest/)
2. **man pages**：[man7.org](https://man7.org/linux/man-pages/) - 系统调用文档
3. **libbpf文档**：[ebpf.io](https://ebpf.io/) - eBPF官方文档
4. **容器运行时接口**：[opencontainers.org](https://www.opencontainers.org/) - OCI规范

### 在线课程
1. **MIT 6.S081**：[操作系统工程](https://pdos.csail.mit.edu/6.828/2023/) - 实现OS
2. **CS140**：[操作系统](https://web.stanford.edu/class/cs140/) - Stanford
3. **OSTEP**：[操作系统导论](http://www.ostep.org/) - 免费在线教材
4. **Linux内核开发**：[LFD420](https://training.linuxfoundation.org/resources/free/) - Linux基金会

### 技术博客与案例
1. **Linux内核公告**：[lkml.org](https://lkml.org/) - 内核邮件列表
2. **Brendan Gregg Blog**：[brendangregg.com](https://www.brendangregg.com/blog/) - 性能分析
3. **eBPF文档**：[ebpf.io](https://ebpf.io/) - eBPF资源
4. **容器技术博客**：[cgroups.md](https://cgroups.md/) - Cgroup详解

### 开源项目参考
1. **Linux内核**：[github.com/torvalds/linux](https://github.com/torvalds/linux) - 内核源码
2. **runc**：[github.com/opencontainers/runc](https://github.com/opencontainers/runc) - OCI运行时
3. **containerd**：[github.com/containerd/containerd](https://github.com/containerd/containerd) - 容器管理
4. **BCC**：[github.com/iovisor/bcc](https://github.com/iovisor/bcc) - eBPF编译器
5. **libbpf**：[github.com/libbpf/libbpf](https://github.com/libbpf/libbpf) - eBPF库
6. **strace**：[github.com/strace/strace](https://github.com/strace/strace) - 系统调用追踪

### 权威论文
1. **UNIX Implementation** (Ken Thompson, 1978)
2. **The Nucleus of a Multiprogramming System** (Dijkstra, 1968)
3. **The Linux Scheduler** (Ingo Molnár, 2007)
4. **Cgroups** (Google, 2006)
5. **Namespaces** (Linux Kernel, 2002)
6. **eBPF** (Jay Schulist, 2014)

### 实用工具
1. **系统调用追踪**：
   - strace（系统调用追踪）
   - ltrace（库函数追踪）
   - ftrace（内核函数追踪）

2. **性能分析**：
   - perf（性能分析）
   - top/htop（进程监控）
   - vmstat（虚拟内存统计）
   - iostat（I/O统计）

3. **内核调试**：
   - crash（内核崩溃分析）
   - dmesg（内核日志）
   - /proc（内核信息）
   - sysfs（内核对象）

## 学习产出要求

### 设计产出
1. ✅ 操作系统内核架构设计文档
2. ✅ 容器运行时设计方案
3. ✅ 进程调度方案设计
4. ✅ 内存管理方案设计
5. ✅ 文件系统选型文档

### 代码产出
1. ✅ 内核模块示例代码
2. ✅ 简易容器运行时实现
3. ✅ 进程池实现
4. ✅ 线程池实现
5. ✅ 共享内存通信实现
6. ✅ eBPF监控程序

### 技能验证
1. ✅ 理解进程与线程的区别
2. ✅ 掌握进程间通信机制
3. ✅ 理解虚拟内存原理
4. ✅ 掌握Linux文件系统
5. ✅ 理解容器原理（Namespace、Cgroup）
6. ✅ 能够编写内核模块
7. ✅ 能够使用eBPF进行性能分析
8. ✅ 能够进行系统调优

### 文档产出
1. ✅ 内核机制总结文档
2. ✅ 容器技术原理解析
3. ✅ 系统调优参数配置
4. ✅ eBPF工具使用指南

## 时间安排建议

### 第1天（进程与内存）
- **上午（4小时）**：进程与线程
  - 进程概念与控制
  - 线程API与同步
  - 实践：多进程/多线程编程

- **下午（4小时）**：进程调度与并发
  - 调度算法
  - 线程同步机制
  - 无锁编程
  - 实践：生产者-消费者模型

- **晚上（2小时）**：内存管理
  - 虚拟内存原理
  - 内存分配器
  - 实践：分析内存布局

### 第2天（文件系统与内核）
- **上午（4小时）**：文件系统
  - VFS架构
  - 文件系统类型
  - 实践：实现简单文件系统

- **下午（4小时）**：进程间通信
  - 管道、消息队列、共享内存
  - Unix Domain Socket
  - 实践：IPC通信实现

- **晚上（2小时）**：内核机制
  - 内核架构
  - 内核模块编程
  - 实践：编写内核模块

### 第3天（容器与eBPF）
- **上午（4小时）**：容器原理
  - Namespace隔离
  - Cgroup限制
  - 实践：创建命名空间

- **下午（4小时）**：eBPF与内核追踪
  - eBPF原理
  - bpftrace工具
  - 实践：编写eBPF程序

- **晚上（2小时）**：容器运行时实现
  - 整合Namespace与Cgroup
  - 实现简单容器
  - 总结与复习

## 学习方法建议

### 1. 理论与实践结合
- 理解原理 → 编写代码 → 运行验证 → 深入理解
- 使用strace追踪系统调用
- 阅读/proc虚拟文件系统
- 编写内核模块验证理论

### 2. 从简单到复杂
- 先理解基本概念（进程、线程）
- 再学习高级机制（Namespace、Cgroup）
- 最后深入实现（容器运行时）

### 3. 源码阅读
- 阅读Linux内核源码
- 阅读glibc实现
- 阅读容器运行时代码

### 4. 工具辅助
- 使用strace追踪系统调用
- 使用perf分析性能
- 使用ftrace追踪内核函数
- 使用eBPF动态监控

### 5. 与其他路径协同
- 与网络编程路径：Socket实现
- 与性能优化路径：系统性能分析
- 与云原生路径：容器深入理解

## 常见问题与解决方案

### Q1：进程 vs 线程？
**A**：核心区别：
- **进程**：独立的地址空间，开销大
- **线程**：共享地址空间，开销小
- **选择**：CPU密集用多进程，IO密集用多线程

### Q2：协程 vs 线程？
**A**：对比分析：
- **协程**：用户态调度，开销极小，无并发
- **线程**：内核态调度，有并发，开销较大
- **选择**：高并发场景用协程（Go、Lua）

### Q3：虚拟内存有什么用？
**A**：核心价值：
- 内存隔离（进程独立）
- 内存保护（防止越界）
- 内存超额（物理内存可以小于虚拟内存）
- 内存映射（文件映射、共享内存）

### Q4：容器 vs 虚拟机？
**A**：对比分析：
- **容器**：共享内核，轻量级，秒级启动
- **虚拟机**：独立内核，隔离性强，分钟级启动
- **选择**：应用隔离用容器，强隔离用虚拟机

### Q5：eBPF vs 内核模块？
**A**：对比分析：
- **eBPF**：安全、可编程、无需重新编译内核
- **内核模块**：功能强大、可修改内核、有安全风险
- **推荐**：监控追踪用eBPF，功能扩展用内核模块

### Q6：如何调试内核问题？
**A**：调试工具链：
1. **dmesg**：查看内核日志
2. **crash**：分析内核崩溃
3. **ftrace**：追踪内核函数
4. **perf**：性能分析
5. **eBPF**：动态追踪

### Q7：如何优化系统性能？
**A**：优化方向：
1. **CPU**：调整调度策略、CPU亲和性
2. **内存**：调整VM参数、使用Huge Pages
3. **I/O**：调整I/O调度算法、使用SSD
4. **网络**：调整TCP参数、启用零拷贝

## 知识体系构建

### 核心知识领域

#### 1. 进程管理
```
进程管理
├── 进程概念
│   ├── 进程定义
│   ├── 进程状态
│   ├── 进程控制块（PCB）
│   └── 进程上下文切换
├── 进程控制
│   ├── fork（创建进程）
│   ├── exec（加载程序）
│   ├── wait（等待子进程）
│   └── exit（终止进程）
├── 进程调度
│   ├── FCFS（先来先服务）
│   ├── SJF（短作业优先）
│   ├── 时间片轮转
│   └── CFS（完全公平调度）
└── 进程通信
    ├── 管道
    ├── 消息队列
    ├── 共享内存
    └── Unix Domain Socket
```

#### 2. 内存管理
```
内存管理
├── 虚拟内存
│   ├── 虚拟地址 vs 物理地址
│   ├── 页表（Page Table）
│   ├── TLB（Translation Lookaside Buffer）
│   └── 缺页中断（Page Fault）
├── 内存分配
│   ├── 分页管理（Paging）
│   ├── 分段管理（Segmentation）
│   └── 段页式管理
├── 内存分配器
│   ├── malloc/free
│   ├── jemalloc
│   └── tcmalloc
└── 内存优化
    ├── 大页内存（Huge Pages）
    ├── 内存池（Memory Pool）
    └── 对象池（Object Pool）
```

#### 3. 文件系统
```
文件系统
├── VFS（虚拟文件系统）
│   ├── super_block
│   ├── inode
│   ├── dentry
│   └── file
├── 文件系统类型
│   ├── ext4（扩展文件系统）
│   ├── xfs（高性能日志文件系统）
│   ├── btrfs（写时复制）
│   └── zfs（企业级）
├── 日志文件系统
│   ├── 日志原理
│   ├── 日志恢复
│   └── 写时复制
└── 分布式文件系统
    ├── NFS
    ├── Ceph
    └── HDFS
```

#### 4. 容器技术
```
容器技术
├── Namespace（命名空间）
│   ├── UTS（主机名）
│   ├── PID（进程ID）
│   ├── Mount（挂载点）
│   ├── Network（网络）
│   ├── User（用户）
│   └── Cgroup（控制组）
├── Cgroup（控制组）
│   ├── CPU限制
│   ├── 内存限制
│   └── I/O限制
├── 容器运行时
│   ├── runc（OCI运行时）
│   ├── containerd（管理）
│   └── CRI-O（K8s运行时）
└── 容器网络
    ├── veth pair
    ├── bridge
    ├── overlay
    └── CNI
```

### 学习深度建议

#### 精通级别
- 进程与线程原理
- 进程间通信（IPC）
- 虚拟内存管理
- Linux文件系统（VFS、ext4）
- Namespace与Cgroup
- 系统调用API

#### 掌握级别
- 进程调度算法
- 内存分配器（malloc、jemalloc）
- 日志文件系统
- 容器运行时（runc、containerd）
- 内核模块编程

#### 了解级别
- 内核源码结构
- 内核同步机制
- eBPF高级应用
- 分布式文件系统
- 内核性能优化

## 下一步学习

### 立即进入
1. **网络编程**（路径11）：
   - Socket底层实现
   - 协同效应：本路径的系统调用 + 网络路径的协议

2. **系统性能优化**（路径14）：
   - 系统性能Profiling
   - 协同效应：本路径的内核知识 + 性能路径的工具

3. **实践项目**：
   - Agent基础设施项目（docs/practice-projects/agent-infrastructure.md）

### 后续深入
1. **计算机组成**（路径15）：理解硬件层
2. **云原生进阶**（路径06）：容器编排
3. **异构超算**（docs/practice-projects/heterogeneous-computing.md）：GPU编程

### 持续跟进
- Linux内核新版本特性
- eBPF技术发展
- 容器技术演进（Kata Containers、gVisor）
- 内核性能优化技术

---

## 学习路径特点

### 针对人群
- 有C语言基础，需要深入理解操作系统
- 面向JD中的"深刻理解操作系统"要求
- 适合需要实现容器/虚拟化的工程师

### 学习策略
- **高强度**：3天集中学习，每天8-10小时
- **重实践**：60%时间动手实践，40%理论学习
- **JD导向**：聚焦容器、内核机制等JD要求
- **底层导向**：深入理解系统底层原理

### 协同学习
- 与网络编程路径：理解Socket底层实现
- 与性能优化路径：学习内核Profiling
- 与云原生路径：深入理解容器原理

### 质量保证
- 所有资源都是权威、最新
- 代码示例可直接运行
- 内核模块可加载测试
- 容器实现可实际使用

---

*学习路径设计：针对有编程基础的工程师，深入理解操作系统内核原理*
*时间窗口：春节第2周后半段3天，高强度学习操作系统内核*
*JD对标：满足JD中操作系统、容器、资源隔离等核心要求*
