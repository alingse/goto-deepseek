# 网络编程与高性能IO（3天）

## 概述
- **目标**：系统掌握网络编程的核心原理与高性能IO技术，深入理解TCP/IP协议栈、IO多路复用、零拷贝等技术，满足JD中"深刻理解计算机网络等核心原理"和"面向数千万日活用户的产品后端架构设计"的要求
- **时间**：春节第2周后半段（3天）
- **前提**：熟悉基本编程语言，了解Socket基础概念
- **强度**：高强度（每天8-10小时），适合需要快速提升网络编程能力的工程师

## JD要求对应

### JD领域覆盖
| JD领域 | 对应内容 | 优先级 |
|--------|----------|--------|
| 一、高并发服务端与API系统 | 高性能网络服务器、C10K/C100K问题、RPC框架 | ⭐⭐⭐ |
| 二、大规模数据处理Pipeline | 网络数据传输优化、零拷贝技术、RDMA | ⭐⭐⭐ |
| 三、Agent基础设施与运行时平台 | 容器网络、服务网格通信、高性能RPC | ⭐⭐ |
| 四、异构超算基础设施 | RDMA高速网络、GPU Direct、集群通信优化 | ⭐⭐ |

### JD能力对应
| 能力要求 | 学习内容 | 验证方式 |
|----------|----------|----------|
| **深刻理解计算机网络** | TCP/IP协议栈、IO模型、网络优化 | 网络服务器实现 |
| **高并发架构设计** | Reactor/Proactor模式、事件驱动 | 高性能服务器设计 |
| **性能优化能力** | 零拷贝、DPDK、内核调优 | 性能测试报告 |
| **分布式系统通信** | RPC框架、消息序列化、服务发现 | RPC框架实现 |

## 学习重点

### 1. TCP/IP协议栈深度解析（第1天上午）
**JD引用**："深刻理解计算机组成、操作系统、计算机网络等核心原理"

**核心内容**：
- OSI七层模型与TCP/IP四层模型
- TCP协议详解
  - 连接管理（三次握手、四次挥手）
  - 序列号与确认号机制
  - 滑动窗口与流量控制
  - 拥塞控制算法（慢启动、拥塞避免、快重传、快恢复）
  - Keep-Alive与心跳机制
  - Nagle算法与延迟ACK
- UDP协议详解
  - UDP vs TCP对比
  - UDP应用场景
  - QUIC协议（基于UDP的可靠传输）
- IP协议与路由
  - IP地址与子网掩码
  - 路由选择算法
  - IPv6新特性
- Socket地址结构与字节序
  - struct sockaddr与sockaddr_in
  - 网络字节序（大端）与主机字节序
  - 地址转换函数

**实践任务**：
- 使用tcpdump抓包分析TCP连接建立过程
- 实现TCP客户端与服务器
- 分析TCP拥塞控制算法行为
- 对比TCP与UDP的性能差异

### 2. Socket编程基础（第1天下午）
**JD引用**："面向数千万日活用户的产品后端架构设计"

**核心内容**：
- Berkeley Socket API
  - socket()创建套接字
  - bind()绑定地址
  - listen()监听连接
  - accept()接受连接
  - connect()发起连接
  - send()/recv()发送接收数据
  - sendto()/recvfrom() UDP数据报
  - closesocket()关闭套接字
- Socket选项配置
  - SO_REUSEADDR端口复用
  - SO_REUSEPORT端口重用
  - SO_KEEPALIVE保活机制
  - SO_LINGER关闭行为
  - SO_SNDBUF/SO_RCVBUF缓冲区大小
  - TCP_NODELAY禁用Nagle算法
- 阻塞IO vs 非阻塞IO
  - fcntl()设置非阻塞模式
  - 非阻塞IO的EAGAIN/EWOULDBLOCK处理
- 同步IO vs 异步IO
  - POSIX AIO
  - Linux io_uring

**实践任务**：
- 实现Echo服务器（TCP）
- 实现简单的HTTP服务器
- 配置Socket选项优化性能
- 实现心跳检测机制

### 3. IO多路复用（第1天晚上）
**JD引用**："面向数千万日活用户的产品后端架构设计"

**核心内容**：
- select()
  - select原理与实现
  - fd_set结构与FD宏
  - select的缺点（1024限制、性能问题）
- poll()
  - poll原理与实现
  - struct pollfd结构
  - poll vs select对比
- epoll（Linux）
  - epoll_create/epoll_create1
  - epoll_ctl注册事件
  - epoll_wait等待事件
  - epoll的两种触发模式（LT/ET）
  - epoll优势（O(1)复杂度、无连接数限制）
- kqueue（BSD/macOS）
  - kqueue原理与API
  - eventfd与kevent结构
- io_uring（Linux 5.1+）
  - io_uring原理（共享队列、零拷贝）
  - io_uring_enter系统调用
  - Submission Queue与Completion Queue

**实践任务**：
- 使用select实现并发服务器
- 使用epoll实现高性能服务器
- 对比select/poll/epoll的性能
- 实现epoll ET模式服务器

### 4. Reactor与Proactor模式（第2天上午）
**JD引用**："能够设计高可用、高可靠的系统架构"

**核心内容**：
- Reactor模式
  - Reactor组件（Handle、Event Demultiplexer、Dispatcher、Handler）
  - 单Reactor单线程
  - 单Reactor多线程
  - 主从Reactor（Multiple Reactors）
  - Reactor模式实现（Netty、libuv、ACE）
- Proactor模式
  - Proactor组件（Asynchronous Operation Processor、Completion Handler）
  - Proactor vs Reactor对比
  - AIO与Proactor的关系
- 事件驱动架构
  - 事件循环（Event Loop）
  - 回调函数设计
  - 优先级队列与定时器
- 高性能网络框架
  - Netty（Java）
  - libuv（C/C++）
  - libevent（C）
  - Twisted（Python）
  - Go netpoll

**实践任务**：
- 实现单Reactor单线程服务器
- 实现主从Reactor多线程服务器
- 使用libevent实现Echo服务器
- 对比不同事件库的性能

### 5. 零拷贝与高性能技术（第2天下午）
**JD引用**："持续优化数据处理各环节的性能与吞吐，确保数据管道的稳定高效"

**核心内容**：
- 传统数据传输的问题
  - read/write系统调用的数据拷贝
  - 4次拷贝（磁盘→内核→用户→内核→网卡）
  - 4次上下文切换
- 零拷贝技术
  - mmap()内存映射
  - sendfile()零拷贝发送
  - splice()管道传输
  - tee()镜像传输
  - vmsplice()用户空间传输
- 零拷贝应用
  - Kafka使用sendfile优化
  - Nginx零拷贝配置
  - 文件服务器优化
- DPDK（Data Plane Development Kit）
  - DPDK原理（轮询模式、UIO/VFIO）
  - PMD驱动模式
  - HugePage内存管理
  - DPDK应用场景
- RDMA（Remote Direct Memory Access）
  - RDMA原理（绕过内核、零拷贝）
  - InfiniBand vs RoCE vs iWARP
  - RDMA编程（libibverbs、rdma-core）
  - RDMA应用场景（存储集群、HPC）

**实践任务**：
- 对比传统send与sendfile性能
- 使用mmap实现文件共享
- 配置Nginx零拷贝优化
- 实现简单的DPDK应用（如有环境）

### 6. C10K/C100K/C1000K问题（第2天晚上）
**JD引用**："深度参与面向数千万日活用户的产品后端架构设计"

**核心内容**：
- C10K问题（1万并发连接）
  - 问题背景与挑战
  - 解决方案（IO多路复用、非阻塞IO）
- C100K问题（10万并发连接）
  - 新的挑战（内存、CPU、文件描述符）
  - 解决方案（连接池、内存优化、epoll）
- C1000K问题（100万并发连接）
  - 极限挑战
  - 解决方案
    - 内核调优（fs.file-max、net.ipv4.ip_local_port_range）
    - 协议优化（TCP Fast Open、TCP Keepalive调优）
    - 应用层优化（连接复用、长连接）
    - eBPF与XDP加速
- 连接池设计
  - 短连接 vs 长连接
  - 连接池大小计算
  - 连接保活与健康检查
  - DNS缓存与连接复用

**实践任务**：
- 配置系统支持100万连接
- 实现高性能连接池
- 压测服务器并发连接数
- 优化内核参数提升并发能力

### 7. RPC框架原理与实践（第3天上午）
**JD引用**："负责核心服务的性能优化、数据库调优与分布式系统可靠性保障"

**核心内容**：
- RPC基础概念
  - RPC vs RESTful对比
  - RPC调用流程（客户端存根→网络传输→服务端存根→服务执行）
  - 同步调用 vs 异步调用
- 序列化协议
  - JSON（文本格式、可读性好）
  - Protobuf（二进制、高效）
  - MessagePack（二进制、紧凑）
  - FlatBuffers（零拷贝、高性能）
  - Cap'n Proto（零拷贝、极致性能）
- 传输协议
  - HTTP/1.1（文本协议）
  - HTTP/2（二进制帧、多路复用）
  - HTTP/3（QUIC、UDP）
  - 自定义TCP协议
- RPC框架实现
  - gRPC（HTTP/2 + Protobuf）
  - Thrift（Facebook）
  - Dubbo（阿里巴巴）
  - Finagle（Twitter）
  - brpc（百度）
- 服务治理
  - 服务注册与发现
  - 负载均衡策略
  - 熔断与降级
  - 超时与重试
  - 服务监控与追踪

**实践任务**：
- 使用Protobuf定义服务接口
- 实现简单的RPC框架
- 集成gRPC实现服务调用
- 实现服务注册中心

### 8. 网络性能优化与调试（第3天下午）
**JD引用**："熟练运用Profiling和可观测性工具分析与定位复杂系统问题"

**核心内容**：
- 网络性能指标
  - 带宽（Bandwidth）
  - 延迟（Latency）
  - 吞吐量（Throughput）
  - 丢包率（Packet Loss）
  - 连接数（Connections）
- 网络性能测试工具
  - iperf/iperf3（带宽测试）
  - netperf（网络性能基准）
  - wrk/ab（HTTP压测）
  - tcpdump/Wireshark（抓包分析）
- 网络调优参数
  - TCP缓冲区（net.ipv4.tcp_rmem/wmem）
  - TCP连接跟踪（net.netfilter.nf_conntrack_max）
  - TIME_WAIT优化（net.ipv4.tcp_tw_reuse）
  - keepalive调优（net.ipv4.tcp_keepalive_*）
- 网络问题排查
  - 网络延迟分析（ping、traceroute、mtr）
  - 连接状态分析（ss、netstat、lsof）
  - 网络错误分析（/proc/net/snmp）
  - 网络监控（Prometheus Node Exporter）
- 高级网络技术
  - eBPF网络监控
  - XDP（eXpress Data Path）
  - TLS加速（OpenSSL offload）

**实践任务**：
- 使用tcpdump分析网络流量
- 使用iperf测试网络带宽
- 调优系统网络参数
- 使用eBPF监控网络性能

## 实践项目：高性能网络服务器

### 项目目标
**JD对应**：满足"面向数千万日活用户的产品后端架构设计"要求

实现一个生产级高性能网络服务器，支持：
1. 基于epoll的高并发连接（支持100万连接）
2. Reactor模式（主从Reactor多线程）
3. 零拷贝文件传输
4. 自定义协议支持
5. 连接池管理
6. 性能监控接口

### 技术栈参考（明确版本）
- **编程语言**：C/C++（C++17）或 Go 1.21+
- **IO模型**：epoll（Linux）、kqueue（macOS/BSD）
- **事件库**：libevent 2.1+ 或 原生epoll
- **序列化**：Protobuf 3.21+ / MessagePack 3.0+
- **监控**：Prometheus 2.45+ + Grafana 10.0+
- **压测**：wrk 4.2+ / iperf3

### 环境配置要求
- **操作系统**：Linux 5.15+（推荐Ubuntu 22.04或CentOS 8+）
- **编译工具**：GCC 11+ / Clang 14+
- **依赖安装**：
  ```bash
  # 安装依赖
  sudo apt-get install libevent-dev libprotobuf-dev protobuf-compiler

  # 安装监控工具
  sudo apt-get install prometheus-node-exporter grafana

  # 安装压测工具
  sudo apt-get install wrk iperf3 tcpdump wireshark

  # 内核调优
  sudo sysctl -w fs.file-max=1000000
  sudo sysctl -w net.ipv4.tcp_max_syn_backlog=8192
  ```

### 架构设计
```
high-performance-server/
├── core/                  # 核心组件
│   ├── reactor/          # Reactor模式实现
│   │   ├── reactor.h/.cpp       # Reactor基类
│   │   ├── main_reactor.h/.cpp  # 主Reactor（accept）
│   │   └── sub_reactor.h/.cpp   # 子Reactor（IO）
│   ├── poller/           # IO多路复用封装
│   │   ├── poller.h/.cpp        # Poller基类
│   │   ├── epoll_poller.h/.cpp  # Epoll实现
│   │   └── kqueue_poller.h/.cpp # Kqueue实现
│   ├── channel/          # 通道封装
│   │   ├── channel.h/.cpp       # Channel类
│   │   └── event_loop.h/.cpp    # EventLoop类
│   └── buffer/           # 缓冲区管理
│       ├── buffer.h/.cpp        # Buffer类
│       └── zero_copy.h/.cpp     # 零拷贝封装
├── net/                  # 网络层
│   ├── socket/           # Socket封装
│   │   ├── socket.h/.cpp        # Socket类
│   │   └── socket_opt.h/.cpp    # Socket选项
│   ├── address/          # 地址封装
│   │   └── inet_address.h/.cpp  # InetAddress类
│   └── protocol/         # 协议定义
│       ├── protocol.h           # 协议基类
│       ├── http_protocol.h      # HTTP协议
│       └── custom_protocol.h    # 自定义协议
├── server/               # 服务器实现
│   ├── tcp_server.h/.cpp        # TCP服务器
│   ├── http_server.h/.cpp       # HTTP服务器
│   └── rpc_server.h/.cpp        # RPC服务器
├── client/               # 客户端实现
│   ├── tcp_client.h/.cpp        # TCP客户端
│   ├── http_client.h/.cpp       # HTTP客户端
│   └── rpc_client.h/.cpp        # RPC客户端
├── codec/                # 编解码
│   ├── protobuf_codec.h/.cpp    # Protobuf编解码
│   ├── json_codec.h/.cpp        # JSON编解码
│   └── msgpack_codec.h/.cpp     # MessagePack编解码
├── pool/                 # 对象池
│   ├── connection_pool.h/.cpp   # 连接池
│   ├── thread_pool.h/.cpp       # 线程池
│   └── memory_pool.h/.cpp       # 内存池
├── metrics/              # 监控指标
│   ├── metrics.h/.cpp           # 指标收集
│   └── prometheus_exporter.cpp  # Prometheus导出
└── examples/             # 示例程序
    ├── echo_server.cpp          # Echo服务器
    ├── http_server.cpp          # HTTP服务器
    ├── file_server.cpp          # 文件服务器（零拷贝）
    └── rpc_server.cpp           # RPC服务器
```

### 核心组件设计

#### 1. Reactor模式实现
```cpp
// 主从Reactor架构
class MainReactor {
    // 负责accept新连接
    // 将新连接分配给SubReactor
    std::unique_ptr<Poller> poller_;
    std::vector<std::unique_ptr<SubReactor>> sub_reactors_;
    int next_reactor_idx_ = 0;  // 轮询分配
};

class SubReactor {
    // 负责已建立连接的IO事件
    // 每个SubReactor运行在独立线程
    std::unique_ptr<Poller> poller_;
    std::unordered_map<int, Channel*> channels_;
    std::thread thread_;
};
```

#### 2. 零拷贝实现
```cpp
// sendfile零拷贝发送
ssize_t zero_copy_send(int out_fd, int in_fd, off_t* offset, size_t count) {
    return sendfile(out_fd, in_fd, offset, count);
}

// splice管道传输
ssize_t splice_pipe(int fd_in, int fd_out, size_t len) {
    int pipefd[2];
    pipe(pipefd);
    splice(fd_in, NULL, pipefd[1], NULL, len, SPLICE_F_MOVE);
    splice(pipefd[0], NULL, fd_out, NULL, len, SPLICE_F_MOVE);
    close(pipefd[0]);
    close(pipefd[1]);
}
```

#### 3. 连接池管理
```cpp
class ConnectionPool {
    struct Connection {
        int fd;
        InetAddress peer_addr;
        time_t last_active;
        bool is_idle;
    };

    std::unordered_map<int, Connection> connections_;
    std::queue<Connection*> idle_connections_;
    size_t max_connections_;

    // 连接保活
    void keepalive_check();

    // 连接超时清理
    void cleanup_idle_connections();
};
```

#### 4. 性能监控
```cpp
class Metrics {
    // 连接数
    std::atomic<uint64_t> connections_{0};
    std::atomic<uint64_t> total_connections_{0};

    // 请求统计
    std::atomic<uint64_t> requests_{0};
    std::atomic<uint64_t> requests_per_sec_{0};

    // 延迟统计
    std::atomic<uint64_t> avg_latency_us_{0};
    std::atomic<uint64_t> p99_latency_us_{0};

    // 导出Prometheus格式
    std::string export_prometheus();
};
```

## 学习资源

### 经典书籍
1. **《UNIX网络编程》**（卷1、卷2）：W. Richard Stevens - 网络编程圣经
2. **《TCP/IP详解》**（卷1、卷2、卷3）：TCP/IP协议权威指南
3. **《Linux高性能服务器编程》**：游双 - 实战导向
4. **《Unix环境高级编程》（APUE）**：网络编程基础
5. **《Boost.Asio C++ Network Programming》**：C++网络编程实战

### 官方文档
1. **Linux man pages**：[socket(7), tcp(7), epoll(7)](https://man7.org/linux/man-pages/)
2. **io_uring文档**：[liburing](https://github.com/axboe/liburing)
3. **DPDK文档**：[DPDK Programmer's Guide](https://doc.dpdk.org/guides/)
4. **gRPC文档**：[gRPC Documentation](https://grpc.io/docs/)

### 在线课程
1. **Stanford CS144**：[计算机网络](https://cs144.github.io/) - 实现TCP协议栈
2. **KU《计算机网络》**：[Coursera课程](https://www.coursera.org/learn/computer-networking)
3. **Linux网络编程**：[实验楼网络编程](https://www.lanqiao.cn/courses/)

### 技术博客与案例
1. **The C10K problem**：[经典问题分析](http://www.kegel.com/c10k.html)
2. **Redis源码分析**：[ae事件循环](https://github.com/redis/redis/tree/unstable/src)
3. **Nginx架构**：[模块化架构设计](https://www.nginx.com/blog/)
4. **Netty原理**：[零拷贝与内存管理](https://netty.io/wiki/user-guide.html)
5. **Cloudflare Blog**：[网络性能优化](https://blog.cloudflare.com/)

### 开源项目参考
1. **libevent**：[github.com/libevent/libevent](https://github.com/libevent/libevent) - 事件通知库
2. **libuv**：[github.com/libuv/libuv](https://github.com/libuv/libuv) - 跨平台异步IO
3. **Netty**：[github.com/netty/netty](https://github.com/netty/netty) - Java网络框架
4. **Muduo**：[github.com/chenshuo/muduo](https://github.com/chenshuo/muduo) - C++网络库
5. **Brpc**：[github.com/apache/brpc](https://github.com/apache/brpc) - 百度RPC框架
6. **gRPC**：[github.com/grpc/grpc](https://github.com/grpc/grpc) - Google RPC框架

### 权威论文
1. **C10K**：[The C10K problem](http://www.kegel.com/c10k.html) (Dan Kegel, 1999)
2. **IO多路复用**：[select/poll/epoll对比](https://man7.org/linux/man-pages/man7/epoll.7.html)
3. **零拷贝**：[Efficient data transfer through zero copy](https://www.kernel.org/doc/Documentation/networking/zero-copy.txt)
4. **RDMA**：[Remote Direct Memory Access](https://www.rdmaconsortium.org/)

### 实用工具
1. **网络测试**：
   - iperf3（带宽测试）
   - netperf（网络性能基准）
   - wrk（HTTP压测）
   - ab（Apache Bench）

2. **抓包分析**：
   - tcpdump（命令行抓包）
   - Wireshark（图形化分析）
   - tshark（Wireshark CLI）

3. **性能监控**：
   - ss（socket统计）
   - netstat（网络状态）
   - lsof（打开文件列表）
   - strace（系统调用追踪）

## 学习产出要求

### 设计产出
1. ✅ 高性能网络服务器架构设计文档
2. ✅ Reactor模式设计方案（主从Reactor）
3. ✅ 零拷贝技术应用方案
4. ✅ RPC框架设计方案
5. ✅ 连接池管理方案
6. ✅ 网络性能优化方案

### 代码产出
1. ✅ 基于epoll的高并发服务器
2. ✅ Reactor模式实现（主从Reactor）
3. ✅ 零拷贝文件传输实现
4. ✅ 简单RPC框架实现
5. ✅ HTTP服务器实现
6. ✅ 连接池实现
7. ✅ 性能监控模块

### 技能验证
1. ✅ 理解TCP/IP协议栈原理
2. ✅ 掌握Socket编程API
3. ✅ 精通IO多路复用（select/poll/epoll）
4. ✅ 理解Reactor/Proactor模式
5. ✅ 掌握零拷贝技术
6. ✅ 能够设计高性能网络服务器
7. ✅ 能够进行网络性能调优
8. ✅ 掌握RPC框架原理

### 文档产出
1. ✅ 网络服务器技术选型文档
2. ✅ 性能测试报告（C10K/C100K）
3. ✅ 内核调优参数配置
4. ✅ 网络问题排查手册

## 时间安排建议

### 第1天（TCP/IP与Socket编程）
- **上午（4小时）**：TCP/IP协议栈深度解析
  - OSI与TCP/IP模型
  - TCP协议详解（连接、流量控制、拥塞控制）
  - UDP与QUIC协议
  - 实践：tcpdump抓包分析

- **下午（4小时）**：Socket编程基础
  - Berkeley Socket API
  - Socket选项配置
  - 阻塞IO vs 非阻塞IO
  - 实践：实现Echo服务器

- **晚上（2小时）**：IO多路复用入门
  - select/poll原理与实现
  - epoll详解
  - 实践：使用epoll实现并发服务器

### 第2天（高性能IO与架构模式）
- **上午（4小时）**：Reactor与Proactor模式
  - Reactor模式详解
  - Proactor模式详解
  - 事件驱动架构
  - 实践：实现主从Reactor

- **下午（4小时）**：零拷贝与高性能技术
  - 零拷贝原理（mmap、sendfile）
  - DPDK与RDMA
  - 实践：零拷贝文件传输

- **晚上（2小时）**：C10K/C100K问题
  - C10K/C100K/C1000K解决方案
  - 连接池设计
  - 实践：压测服务器性能

### 第3天（RPC框架与性能优化）
- **上午（4小时）**：RPC框架原理与实践
  - RPC基础概念
  - 序列化协议对比
  - gRPC/Thrift/Dubbo
  - 实践：实现简单RPC框架

- **下午（4小时）**：网络性能优化与调试
  - 网络性能指标
  - 性能测试工具
  - 内核参数调优
  - 实践：网络压测与优化

- **晚上（2小时）**：总结与项目实战
  - 高性能服务器实现总结
  - 性能优化技巧总结
  - 制定后续学习计划

## 学习方法建议

### 1. 理论与实践结合（30%理论 + 70%实践）
- 网络编程需要大量实践
- 每个知识点都要动手验证
- 使用tcpdump抓包验证理论
- 编写代码巩固理解

### 2. 从简单到复杂
- 先实现简单的Echo服务器
- 逐步添加功能（非阻塞、epoll、多线程）
- 最终实现完整的高性能服务器
- 对比不同实现的性能差异

### 3. 关注性能瓶颈
- 使用strace分析系统调用
- 使用perf分析CPU性能
- 使用tcpdump分析网络流量
- 定位并优化性能瓶颈

### 4. 阅读优秀开源代码
- Redis事件循环（ae.c）
- Nginx模块化架构
- Netty零拷贝实现
- Muduo网络库设计

### 5. 与其他路径协同
- 与操作系统路径：理解内核网络栈
- 与性能优化路径：学习Profiling工具
- 与分布式系统路径：理解RPC通信

## 常见问题与解决方案

### Q1：select/poll/epoll如何选择？
**A**：选择建议：
- **select**：跨平台兼容性好，但性能差（1024限制）
- **poll**：无连接数限制，但性能随连接数下降
- **epoll**：Linux上性能最优，推荐使用
- **kqueue**：BSD/macOS上使用
- **io_uring**：Linux 5.1+，极致性能

### Q2：阻塞IO vs 非阻塞IO？
**A**：对比分析：
- **阻塞IO**：简单易用，但一个线程只能处理一个连接
- **非阻塞IO**：需要配合IO多路复用，一个线程处理多个连接
- **推荐**：高并发场景使用非阻塞IO + epoll

### Q3：Reactor vs Proactor？
**A**：模式对比：
- **Reactor**：同步IO模型，主动读写（epoll）
- **Proactor**：异步IO模型，完成通知（AIO）
- **推荐**：Linux上使用Reactor（epoll），Windows上使用Proactor（IOCP）

### Q4：如何支持百万并发连接？
**A**：关键优化：
1. **内核调优**：
   ```bash
   fs.file-max = 1000000
   net.ipv4.tcp_max_syn_backlog = 8192
   net.core.somaxconn = 8192
   ```
2. **应用优化**：
   - 使用epoll LT模式
   - 减少系统调用次数
   - 连接池复用
3. **硬件**：足够内存（每连接约4KB）

### Q5：零拷贝为什么快？
**A**：对比分析：
- **传统方式**：4次拷贝 + 4次上下文切换
- **零拷贝**：2次拷贝 + 2次上下文切换
- **性能提升**：减少CPU和内存带宽消耗

### Q6：gRPC vs RESTful？
**A**：场景对比：
- **RESTful**：简单、通用、HTTP/1.1
- **gRPC**：高效、双向流、Protobuf、HTTP/2
- **推荐**：内部服务使用gRPC，外部API使用RESTful

### Q7：如何调试网络问题？
**A**：调试工具链：
1. **连通性**：ping、traceroute、mtr
2. **端口状态**：telnet、nc
3. **抓包分析**：tcpdump、Wireshark
4. **连接状态**：ss、netstat、lsof
5. **性能分析**：iperf3、netperf、wrk

## 知识体系构建

### 核心知识领域

#### 1. 网络协议栈
```
网络协议栈
├── 应用层
│   ├── HTTP/1.1
│   ├── HTTP/2
│   ├── HTTP/3 (QUIC)
│   └── 自定义协议
├── 传输层
│   ├── TCP（可靠传输）
│   ├── UDP（不可靠传输）
│   └── SCTP
├── 网络层
│   ├── IPv4
│   ├── IPv6
│   └── 路由协议
└── 链路层
    ├── 以太网
    ├── WiFi
    └── RDMA
```

#### 2. IO模型
```
IO模型演进
├── 阻塞IO
│   └── 一个线程一个连接
├── 非阻塞IO
│   └── 忙等待，CPU利用率高
├── IO多路复用
│   ├── select（O(n)复杂度）
│   ├── poll（O(n)复杂度）
│   ├── epoll（O(1)复杂度）
│   ├── kqueue（BSD）
│   └── io_uring（异步）
└── 异步IO
    ├── POSIX AIO
    ├── Linux IO_CB
    └── Windows IOCP
```

#### 3. 架构模式
```
网络架构模式
├── Reactor模式
│   ├── 单Reactor单线程
│   ├── 单Reactor多线程
│   └── 主从Reactor（推荐）
├── Proactor模式
│   └── 异步IO模型
└── 事件驱动架构
    ├── 事件循环
    ├── 回调函数
    └── 定时器管理
```

#### 4. 高性能技术
```
高性能技术
├── 零拷贝
│   ├── mmap（内存映射）
│   ├── sendfile（文件传输）
│   ├── splice（管道）
│   └── DMA（直接内存访问）
├── 用户态网络
│   ├── DPDK（轮询模式）
│   ├── XDP（eXpress Data Path）
│   └── PF_RING
├── RDMA
│   ├── InfiniBand
│   ├── RoCE（RDMA over Converged Ethernet）
│   └── iWARP
└── 协议优化
    ├── HTTP/2（多路复用）
    ├── QUIC（UDP-based）
    └── TLS加速
```

### 学习深度建议

#### 精通级别
- TCP/IP协议栈原理（三次握手、四次挥手、拥塞控制）
- Socket编程API（Berkeley Socket）
- IO多路复用（epoll）
- Reactor模式（主从Reactor）
- 零拷贝技术（sendfile、mmap）

#### 掌握级别
- UDP与QUIC协议
- kqueue/io_uring
- Proactor模式
- RPC框架原理
- 网络性能调优

#### 了解级别
- DPDK用户态网络
- RDMA高速网络
- XDP内核加速
- TLS/SSL协议
- SDN软件定义网络

## 下一步学习

### 立即进入
1. **操作系统内核与原理**（路径13）：
   - 理解内核网络栈实现
   - 学习网络协议栈原理
   - 协同效应：本路径的应用层 + 操作系统路径的内核层

2. **系统性能优化与Profiling**（路径14）：
   - 学习网络性能Profiling
   - 掌握性能调优工具
   - 协同效应：本路径的网络优化 + 性能路径的工具方法

3. **实践项目实现**：
   - 高性能API项目（docs/practice-projects/high-performance-api.md）

### 后续深入
1. **分布式系统复习**（路径05）：RPC与分布式通信
2. **云原生进阶**（路径06）：Service Mesh网络
3. **异构超算项目**（docs/practice-projects/heterogeneous-computing.md）：RDMA网络

### 持续跟进
- HTTP/3与QUIC协议发展
- eBPF与XDP技术
- DPDK新版本特性
- RDMA生态发展

---

## 学习路径特点

### 针对人群
- 有C/C++/Go基础，需要学习网络编程
- 面向JD中的"高并发服务端与API系统"要求
- 适合需要实现高性能服务器的工程师

### 学习策略
- **高强度**：3天集中学习，每天8-10小时
- **重实践**：70%时间动手实践，30%理论学习
- **JD导向**：所有学习内容都对应JD要求
- **性能导向**：关注高性能技术与优化

### 协同学习
- 与操作系统路径并行：理解内核网络栈
- 与性能优化路径协同：学习Profiling工具
- 与分布式系统路径互补：RPC与分布式通信

### 质量保证
- 所有资源都是权威、最新
- 代码示例可直接运行
- 性能测试数据真实可验证
- 架构设计文档完整

---

*学习路径设计：针对有编程基础的后端工程师，系统学习网络编程与高性能IO*
*时间窗口：春节第2周后半段3天，高强度快速提升网络编程能力*
*JD对标：满足JD中计算机网络、高并发架构、性能优化等核心要求*
