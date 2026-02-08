# 系统性能优化与Profiling（2天）

## 概述
- **目标**：系统掌握性能分析方法与优化技术，熟练运用各类Profiling工具分析与定位复杂系统问题，满足JD中"熟练运用Profiling和可观测性工具分析与定位复杂系统问题"的要求
- **时间**：春节第3周（2天）
- **前提**：熟悉Linux命令行，了解基本性能概念
- **强度**：高强度（每天8-10小时），适合需要提升性能分析能力的工程师

## JD要求对应

### JD领域覆盖
| JD领域 | 对应内容 | 优先级 |
|--------|----------|--------|
| 一、高并发服务端与API系统 | CPU/内存/IO优化、并发调优、延迟分析 | ⭐⭐⭐ |
| 二、大规模数据处理Pipeline | 数据处理性能、I/O优化、Pipeline调优 | ⭐⭐⭐ |
| 三、Agent基础设施与运行时平台 | 资源利用率、性能监控、故障定位 | ⭐⭐⭐ |
| 四、异构超算基础设施 | GPU性能分析、RDMA优化、集群调度 | ⭐⭐ |

### JD能力对应
| 能力要求 | 学习内容 | 验证方式 |
|----------|----------|----------|
| **Profiling工具** | perf、eBPF、火焰图、pprof | 性能分析报告 |
| **CPU性能优化** | CPU利用率、上下文切换、锁竞争 | 优化方案 |
| **内存性能优化** | 内存泄漏、Cache友好、内存带宽 | 优化方案 |
| **I/O性能优化** | 磁盘I/O、网络I/O、异步I/O | 优化方案 |
| **可观测性** | 指标、日志、追踪、告警 | 监控体系设计 |

## 学习重点

### 1. 性能分析基础（第1天上午）
**JD引用**："熟练运用Profiling和可观测性工具分析与定位复杂系统问题"

**核心内容**：
- 性能指标体系
  - **延迟（Latency）**：请求处理时间
  - **吞吐量（Throughput）**：单位时间处理量
  - **CPU利用率**：CPU忙碌时间占比
  - **内存使用**：RSS、VSZ、堆内存
  - **I/O吞吐**：磁盘读写速度
  - **网络吞吐**：带宽利用率
  - **错误率**：请求失败比例
- 性能分析方法
  - USE方法（Utilization、Saturation、Errors）
  - RED方法（Rate、Errors、Duration）
  - NASA方法（制定SLI/SLO）
- 性能优化方法论
  - 性能金字塔
  - 优化ROI分析
  - 性能测试基线
- 性能分析流程
  1. 明确性能目标
  2. 收集性能数据
  3. 分析瓶颈
  4. 提出优化方案
  5. 验证优化效果

**实践任务**：
- 建立性能指标体系
- 制定SLI/SLO指标
- 使用top/htop观察系统状态
- 使用vmstat分析系统资源

### 2. CPU性能分析（第1天下午）
**JD引用**："熟练运用Profiling和可观测性工具分析与定位复杂系统问题"

**核心内容**：
- CPU性能指标
  - CPU利用率（User/System/Wait/Idle）
  - 上下文切换次数
  - 运行队列长度
  - 中断频率
  - CPU缓存命中率
- CPU分析工具
  - **top/htop**：实时进程监控
  - **mpstat**：多核CPU统计
  - **pidstat**：进程/线程CPU统计
  - **perf**：Linux性能分析工具
    - perf top：热点函数
    - perf record：采样记录
    - perf report：结果分析
    - perf annotate：源码级分析
- CPU性能分析方法
  - 函数级分析（Function Profiling）
  - 指令级分析（Instruction Profiling）
  - 热点路径分析（Hot Path）
  - 锁竞争分析（Lock Contention）
- CPU优化技术
  - CPU亲和性（CPU Affinity）
  - NUMA感知调度
  - 减少上下文切换
  - 减少锁竞争
  - SIMD向量化
  - 编译优化（-O2/-O3）

**实践任务**：
- 使用perf top分析CPU热点
- 使用perf record/report进行采样分析
- 使用火焰图可视化CPU使用
- 配置CPU亲和性优化

### 3. 内存性能分析（第1天晚上）
**JD引用**："负责核心服务的性能优化、数据库调优"

**核心内容**：
- 内存性能指标
  - 物理内存使用（Used/Free/Cached/Buffers）
  - 虚拟内存使用（VSZ/RSS）
  - 页面交换（Swap In/Out）
  - 缺页异常次数
  - 内存分配速率
- 内存分析工具
  - **free**：内存使用概览
  - **vmstat**：虚拟内存统计
  - **smem**：内存使用分析
  - **memleak**：内存泄漏检测
  - **valgrind**：内存调试工具
    - memcheck：内存错误检测
    - massif：堆内存分析
    - callgrind：函数调用分析
- 内存分析方法
  - 内存泄漏检测
  - 内存使用分布
  - 内存分配热点
  - 缓存命中率分析
- 内存优化技术
  - 内存池（Memory Pool）
  - 对象池（Object Pool）
  - 大页内存（Huge Pages）
  - 内存复用
  - 减少内存碎片
  - Cache友好设计

**实践任务**：
- 使用valgrind检测内存泄漏
- 使用memleak分析内存分配
- 使用smem可视化内存使用
- 实现内存池优化

### 4. I/O性能分析（第2天上午）
**JD引用**："持续优化数据处理各环节的性能与吞吐"

**核心内容**：
- I/O性能指标
  - 磁盘I/O（IOPS、吞吐量、延迟）
  - 网络I/O（带宽、延迟、丢包）
  - 缓存命中率
  - 队列长度
- I/O分析工具
  - **iostat**：磁盘I/O统计
  - **iotop**：进程级I/O监控
  - **blktrace**：块设备I/O追踪
  - **bpftrace**：I/O追踪
  - **nethogs**：网络流量监控
  - **iftop**：网络流量分析
- I/O分析方法
  - I/O延迟分布
  - I/O模式分析
  - 热点文件识别
  - 缓存效率分析
- I/O优化技术
  - 异步I/O（AIO、IO_uring）
  - I/O调度算法（CFQ、Deadline、NOOP、mq-deadline）
  - 磁盘优化（SSD、RAID）
  - 文件系统优化（挂载参数、预读）
  - 零拷贝（sendfile、mmap）
  - 缓存策略（多级缓存）

**实践任务**：
- 使用iostat分析磁盘I/O
- 使用blktrace追踪I/O
- 配置异步I/O优化
- 优化文件I/O性能

### 5. 网络性能分析（第2天下午）
**JD引用**："面向数千万日活用户的产品后端架构设计"

**核心内容**：
- 网络性能指标
  - 带宽（Bandwidth）
  - 延迟（Latency）
  - 丢包率（Packet Loss）
  - 连接数
  - TCP重传率
  - TCP连接队列
- 网络分析工具
  - **ss**：Socket统计
  - **netstat**：网络状态
  - **tcpdump**：抓包分析
  - **Wireshark**：协议分析
  - **iperf3**：带宽测试
  - **traceroute**：路由追踪
  - **mtr**：网络诊断
- 网络分析方法
  - TCP连接状态分析
  - 网络延迟分布
  - 丢包原因分析
  - 带宽瓶颈分析
- 网络优化技术
  - TCP参数调优
  - 连接池优化
  - HTTP/2多路复用
  - 协议优化（gRPC、QUIC）
  - 负载均衡
  - CDN加速

**实践任务**：
- 使用tcpdump抓包分析
- 使用ss分析连接状态
- 使用iperf3测试带宽
- 优化TCP参数

### 6. 火焰图分析（第2天下午）
**JD引用**："熟练运用Profiling和可观测性工具"

**核心内容**：
- 火焰图原理
  - 堆栈追踪可视化
  - 宽度表示占比
  - 上下包含关系
- 火焰图类型
  - **CPU火焰图**：CPU时间分布
  - **内存火焰图**：内存分配分布
  - **Off-CPU火焰图**：阻塞时间分布
  - **散点图**：延迟分布
- 火焰图工具
  - **FlameGraph**： Brendan Gregg开源工具集
  - **perf** + **FlameGraph**
  - **bpftrace** + **FlameGraph**
  - **pprof** + **FlameGraph**
- 火焰图分析方法
  - 识别热点函数
  - 分析调用路径
  - 发现优化机会
  - 验证优化效果

**实践任务**：
- 生成CPU火焰图
- 生成内存火焰图
- 分析火焰图定位瓶颈
- 优化后对比火焰图

### 7. 可观测性体系（第2天晚上）
**JD引用**："熟练运用Profiling和可观测性工具分析与定位复杂系统问题"

**核心内容**：
- 可观测性三大支柱
  - **指标（Metrics）**：数值型时序数据
  - **日志（Logs）**：事件记录
  - **追踪（Traces）**：请求链路
- 指标采集体系
  - **Prometheus**：指标采集与存储
    - Pull模式采集
    - PromQL查询语言
    - 服务发现集成
  - **InfluxDB**：时序数据库
  - **OpenTelemetry**：统一可观测性
    - 指标标准化
    - 追踪标准化
    - 日志标准化
- 日志体系
  - **ELK Stack**：Elasticsearch + Logstash + Kibana
  - **EFK Stack**：Elasticsearch + Fluentd + Kibana
  - **Loki**：轻量级日志系统
- 追踪体系
  - **Jaeger**：分布式追踪
  - **Zipkin**：分布式追踪
  - **OpenTelemetry**：统一追踪
- 告警体系
  - **Prometheus AlertManager**
  - **Grafana Alerts**
  - **PagerDuty**

**实践任务**：
- 部署Prometheus监控
- 配置Grafana仪表盘
- 集成OpenTelemetry
- 设置告警规则

## 实践项目：全链路性能监控系统

### 项目目标
**JD对应**：满足"熟练运用Profiling和可观测性工具分析与定位复杂系统问题"要求

设计并实现一个全链路性能监控系统，支持：
1. 基础设施监控（CPU、内存、I/O、网络）
2. 应用性能监控（APM）
3. 分布式追踪
4. 日志聚合
5. 告警与通知

### 技术栈参考（明确版本）
- **指标采集**：Prometheus 2.45+
- **可视化**：Grafana 10.0+
- **日志**：Loki 2.9+ / Elasticsearch 8.x + Kibana 10.x
- **追踪**：Jaeger 1.48+ / Tempo 2.0+
- **APM**：Pyroscope 1.0+ / SkyWalking 9.0+
- **告警**：AlertManager 0.25+
- **语言探针**：OpenTelemetry 1.20+

### 环境配置要求
- **操作系统**：Linux 5.15+（推荐Ubuntu 22.04）
- **依赖**：
  ```bash
  # Docker Compose部署
  curl -L https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose

  # 启动监控系统
  docker-compose -f monitoring.yml up -d
  ```

### 架构设计
```
performance-monitoring/
├── monitoring/               # 监控系统
│   ├── prometheus/           # 指标采集
│   │   ├── prometheus.yml    # 配置文件
│   │   ├── rules/           # 告警规则
│   │   └── targets/         # 监控目标
│   ├── grafana/             # 可视化
│   │   ├── dashboards/      # 仪表盘
│   │   ├── datasources/     # 数据源
│   │   └── alerts/          # 告警配置
│   ├── loki/                 # 日志系统
│   │   ├── loki.yml         # 配置文件
│   │   └── rules/           # 日志告警
│   ├── tempo/               # 追踪存储
│   │   ├── tempo.yml        # 配置文件
│   │   └── querier/         # 查询服务
│   ├── alertmanager/         # 告警管理
│   │   ├── alertmanager.yml  # 配置文件
│   │   └── templates/       # 告警模板
│   └── exporters/           # 指标导出器
│       ├── node-exporter/   # 节点指标
│       ├── cadvisor/        # 容器指标
│       ├── blackbox-exporter/ # 网络探测
│       └── mysqld-exporter/ # MySQL指标
├── apm/                      # 应用性能监控
│   ├── pyroscope/           # Go应用APM
│   ├── skywalking/          # Java应用APM
│   └── opentelemetry-sdk/   # 统一APM SDK
├── tracing/                  # 分布式追踪
│   ├── jaeger/              # Jaeger部署
│   ├── zipkin/              # Zipkin部署
│   └── opentelemetry-col/   # OTel Collector
├── alerts/                   # 告警规则
│   ├── cpu-alerts.yml       # CPU告警
│   ├── memory-alerts.yml    # 内存告警
│   ├── io-alerts.yml       # I/O告警
│   └── latency-alerts.yml   # 延迟告警
├── dashboards/               # Grafana仪表盘
│   ├── node-overview.json    # 节点概览
│   ├── cpu-analysis.json     # CPU分析
│   ├── memory-analysis.json  # 内存分析
│   ├── io-analysis.json      # I/O分析
│   └── network-analysis.json # 网络分析
├── scripts/                  # 脚本工具
│   ├── generate-flamegraph.sh  # 生成火焰图
│   ├── analyze-profile.sh      # 分析性能数据
│   └── setup-monitoring.sh     # 安装监控
└── examples/                 # 示例应用
    ├── golang-app/          # Go应用示例
    ├── python-app/          # Python应用示例
    └── java-app/            # Java应用示例
```

### 核心组件设计

#### 1. Prometheus配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'mysqld'
    static_configs:
      - targets: ['mysqld-exporter:9104']

  - job_name: 'blackbox'
    metrics_path: /probe
    static_configs:
      - targets:
          - http://example.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
```

#### 2. 告警规则配置
```yaml
# rules/cpu-alerts.yml
groups:
  - name: cpu_alerts
    rules:
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighContextSwitches
        expr: rate(node_context_switches_total[5m]) > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High context switches on {{ $labels.instance }}"
```

#### 3. 火焰图生成脚本
```bash
#!/bin/bash
# generate-flamegraph.sh

# 使用perf生成CPU火焰图
PERF_DIR=/tmp/perf
OUTPUT_DIR=/tmp/flamegraph

# 1. 录制perf数据
perf record -F 99 -a -g -- sleep 60

# 2. 生成火焰图
perf script | \
    FlameGraph/stackcollapse-perf.pl | \
    FlameGraph/flamegraph.pl \
    --title="CPU Flame Graph" \
    > $OUTPUT_DIR/cpu.svg

# 3. 生成内存火焰图
perf record -F 99 -a -g -e kmem:kmalloc -- sleep 60

# 4. 使用bpftrace生成内存分配火焰图
bpftrace -e 'profile:hz:99 /comm == "myapp"/ {
    @[ustack] = count();
}' -o $OUTPUT_DIR/memory.bt

# 5. 生成Off-CPU火焰图
bpftrace -e 'sched:sched_switch {
    @[comm, stack] = count();
}' -o $OUTPUT_DIR/offcpu.bt
```

## 学习资源

### 经典书籍
1. **《性能之巅》**：Brendan Gregg - 系统性能权威指南
2. **《Linux性能优化》**：Phillip G. Ezolt
3. **《Web性能权威指南》**：Ilya Grigorik
4. **《数据密集型应用系统设计》**：Martin Kleppmann - 性能章节

### 官方文档
1. **perf Wiki**：[perf wiki](https://perf.wiki.kernel.org/)
2. **FlameGraph**：[github.com/brendangregg/FlameGraph](https://github.com/brendangregg/FlameGraph)
3. **Prometheus**：[prometheus.io/docs](https://prometheus.io/docs/)
4. **Grafana**：[grafana.com/docs](https://grafana.com/docs/)
5. **eBPF**：[ebpf.io/docs](https://ebpf.io/docs/)

### 在线课程
1. **Brendan Gregg Blog**：[brendangregg.com](https://www.brendangregg.com/) - 性能分析
2. **Linux Performance**：[linux-performance.com](http://www.brendangregg.com//linuxperf.html) - 性能工具集
3. **Netflix Tech Blog**：[Netflix性能分析](https://netflixtechblog.com/)
4. **USENIX LISA**：[系统管理会议](https://www.usenix.org/conference/lisa)

### 技术博客与案例
1. **Netflix Tech Blog**：性能优化案例
2. **Facebook Engineering**：性能分析实践
3. **Cloudflare Blog**：网络性能优化
4. **Datadog Blog**：APM实践

### 开源项目参考
1. **FlameGraph**：[github.com/brendangregg/FlameGraph](https://github.com/brendangregg/FlameGraph)
2. **perf-tools**：[github.com/brendangregg/perf-tools](https://github.com/brendangregg/perf-tools)
3. **bpftrace**：[github.com/iovisor/bpftrace](https://github.com/iovisor/bpftrace)
4. **Prometheus**：[github.com/prometheus/prometheus](https://github.com/prometheus/prometheus)
5. **Grafana**：[github.com/grafana/grafana](https://github.com/grafana/grafana)

### 实用工具清单
```
性能分析工具
├── CPU分析
│   ├── perf（Linux性能分析）
│   ├── top/htop（实时监控）
│   ├── mpstat（多核统计）
│   └── pidstat（进程统计）

├── 内存分析
│   ├── valgrind（内存调试）
│   ├── memleak（泄漏检测）
│   ├── smem（内存可视化）
│   └── /proc/meminfo（内存信息）

├── I/O分析
│   ├── iostat（I/O统计）
│   ├── iotop（进程I/O）
│   ├── blktrace（块设备追踪）
│   └── ltrace（库调用追踪）

├── 网络分析
│   ├── tcpdump（抓包）
│   ├── Wireshark（协议分析）
│   ├── ss（Socket统计）
│   └── iperf3（带宽测试）

├── 可视化
│   ├── FlameGraph（火焰图）
│   ├── Grafana（仪表盘）
│   └── pprof（Go性能分析）
```

## 学习产出要求

### 设计产出
1. ✅ 性能指标体系设计（SLI/SLO）
2. ✅ 监控告警方案设计
3. ✅ 全链路监控系统架构
4. ✅ 性能优化方案文档

### 代码产出
1. ✅ 性能监控脚本
2. ✅ 火焰图生成工具
3. ✅ Prometheus配置（告警规则）
4. ✅ Grafana仪表盘
5. ✅ APM集成示例

### 技能验证
1. ✅ 掌握perf性能分析
2. ✅ 掌握火焰图生成与分析
3. ✅ 掌握Prometheus监控
4. ✅ 掌握Grafana可视化
5. ✅ 能够定位性能瓶颈
6. ✅ 能够设计监控体系

### 文档产出
1. ✅ 性能分析方法论
2. ✅ 工具使用手册
3. ✅ 监控配置指南
4. ✅ 优化案例总结

## 时间安排建议

### 第1天（性能基础与分析工具）
- **上午（4小时）**：性能分析基础
  - 性能指标体系
  - 分析方法论
  - 实践：使用top/vmstat

- **下午（4小时）**：CPU与内存分析
  - perf工具详解
  - CPU性能优化
  - 内存泄漏检测
  - 实践：perf record/report

- **晚上（2小时）**：火焰图分析
  - 火焰图原理
  - 生成CPU火焰图
  - 分析火焰图定位瓶颈
  - 实践：生成火焰图

### 第2天（I/O、网络与监控体系）
- **上午（4小时）**：I/O与网络分析
  - iostat/iotop使用
  - 网络性能分析
  - 异步I/O优化
  - 实践：分析I/O瓶颈

- **下午（4小时）**：可观测性体系
  - Prometheus/Grafana
  - 日志聚合
  - 分布式追踪
  - 告警配置
  - 实践：部署监控系统

- **晚上（2小时）**：总结与实践
  - 全链路监控项目
  - 优化案例分析
  - 制定优化计划

## 学习方法建议

### 1. 从指标到追踪
- 先看指标定位问题领域
- 再用追踪定位具体问题
- 最后用代码分析定位行号

### 2. 建立性能基线
- 记录正常性能数据
- 对比异常数据
- 快速识别偏差

### 3. 善用可视化
- 火焰图快速识别热点
- 仪表盘实时监控
- 图表对比优化效果

### 4. 理论与实践结合
- 理解性能原理
- 动手实践工具
- 解决实际问题

## 常见问题与解决方案

### Q1：如何选择性能指标？
**A**：指标选择：
- **系统层**：CPU、内存、I/O、网络
- **应用层**：QPS、延迟、错误率
- **业务层**：转化率、响应时间

### Q2：perf与bpftrace如何选择？
**A**：工具选择：
- **perf**：功能全面，稳定可靠
- **bpftrace**：灵活强大，动态追踪
- **推荐**：perf用于常规分析，bpftrace用于深度分析

### Q3：如何分析CPU瓶颈？
**A**：分析步骤：
1. perf top看热点函数
2. perf record采样
3. 生成火焰图
4. 定位具体代码行

### Q4：如何分析内存泄漏？
**A**：分析方法：
1. top观察内存增长
2. valgrind检测泄漏点
3. 堆内存分析（massif）
4. 长期运行验证

### Q5：如何优化I/O性能？
**A**：优化方向：
1. 使用异步I/O
2. 调整I/O调度算法
3. 增加缓存
4. 使用SSD
5. 零拷贝优化

### Q6：如何设计监控体系？
**A**：设计原则：
1. **指标**：选择关键指标（RED方法）
2. **告警**：设置合理阈值
3. **可视化**：清晰仪表盘
4. **日志**：结构化日志

### Q7：火焰图如何解读？
**A**：解读方法：
- **宽度**表示占比
- **从下到上**是调用栈
- **顶部**是热点函数
- **颜色**没有特殊含义

## 知识体系构建

### 核心知识领域

#### 1. 性能分析方法论
```
性能分析
├── 指标体系
│   ├── 延迟（Latency）
│   ├── 吞吐量（Throughput）
│   ├── 资源利用率
│   └── 错误率
├── 分析方法
│   ├── USE方法（Utilization/Saturation/Errors）
│   ├── RED方法（Rate/Errors/Duration）
│   └── 负载测试
└── 优化流程
    ├── 明确目标
    ├── 收集数据
    ├── 分析瓶颈
    └── 验证优化
```

#### 2. 性能分析工具
```
工具分类
├── CPU分析
│   ├── perf
│   ├── top/htop
│   └── FlameGraph
├── 内存分析
│   ├── valgrind
│   ├── smem
│   └── memleak
├── I/O分析
│   ├── iostat
│   ├── iotop
│   └── blktrace
├── 网络分析
│   ├── tcpdump
│   ├── ss
│   └── iperf3
└── 可视化
    ├── Grafana
    ├── FlameGraph
    └── pprof
```

#### 3. 可观测性体系
```
可观测性
├── 指标（Metrics）
│   ├── Prometheus
│   ├── InfluxDB
│   └── OpenTelemetry
├── 日志（Logs）
│   ├── ELK Stack
│   ├── EFK Stack
│   └── Loki
└── 追踪（Traces）
    ├── Jaeger
    ├── Zipkin
    └── OpenTelemetry
```

### 学习深度建议

#### 精通级别
- perf性能分析
- 火焰图生成与分析
- Prometheus/Grafana监控
- 性能瓶颈定位

#### 掌握级别
- valgrind内存分析
- bpftrace动态追踪
- APM集成
- 告警配置

#### 了解级别
- eBPF高级应用
- 分布式追踪
- 机器学习异常检测
- 云原生监控（Prometheus Operator）

## 下一步学习

### 立即进入
1. **操作系统内核**（路径13）：
   - 深入理解系统调用开销
   - 协同效应：本路径的工具 + 操作系统路径的原理

2. **网络编程**（路径11）：
   - 网络性能优化
   - 协同效应：本路径的网络分析 + 网络编程的优化

### 后续深入
1. **计算机组成**（路径15）：理解硬件性能边界
2. **异构超算**（docs/practice-projects/heterogeneous-computing.md）：GPU性能分析

### 持续跟进
- eBPF性能工具发展
- 可观测性标准演进
- 性能优化新技术
- 云原生监控最佳实践

---

## 学习路径特点

### 针对人群
- 需要掌握性能分析工具的工程师
- 面向JD中的"Profiling和可观测性工具"要求
- 适合需要解决实际性能问题的工程师

### 学习策略
- **高强度**：2天集中学习，每天8-10小时
- **重实践**：70%时间动手实践，30%理论学习
- **工具导向**：聚焦主流性能分析工具
- **实战驱动**：基于真实场景进行性能分析

### 协同学习
- 与操作系统路径：理解底层原理
- 与网络编程路径：分析网络性能
- 与云原生路径：容器性能监控

### 质量保证
- 所有工具都是生产级
- 案例都是真实场景
- 可直接用于工作
- 有完整的监控系统示例

---

*学习路径设计：针对需要提升性能分析能力的工程师，掌握Profiling工具*
*时间窗口：春节第3周2天，高强度学习性能分析与监控*
*JD对标：满足JD中Profiling工具、可观测性、性能优化等核心要求*
