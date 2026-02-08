# 实践项目：Agent运行平台原型

## 项目概述
- **目标**：设计并实现一个简易的AI Agent运行平台原型
- **技术栈**：Python + FastAPI + Kubernetes + Docker + Redis
- **时间**：春节第1周后2天完成
- **关联学习**：Python现代化开发 + 云原生进阶

## 项目背景
随着AI Agent技术的发展，企业需要能够安全、高效地运行和管理大量AI Agent的平台。本项目旨在构建一个简易的Agent运行平台原型，探索容器调度、资源隔离、运行时管理等核心问题，为实际生产环境提供参考。

## 功能需求

### 核心功能
1. **Agent容器管理**：创建、启动、停止、删除Agent容器
2. **资源隔离与限制**：CPU、内存、GPU资源配额控制
3. **调度策略**：基于资源可用性和负载的智能调度
4. **运行时监控**：容器状态、资源使用、性能指标监控
5. **安全沙箱**：网络隔离、文件系统隔离、权限控制

### 高级功能（可选）
1. **弹性伸缩**：根据负载自动扩缩容Agent实例
2. **故障恢复**：Agent故障时自动重启或迁移
3. **版本管理**：Agent镜像版本控制和回滚
4. **多租户支持**：不同用户/团队的资源隔离和配额
5. **计费统计**：资源使用统计和成本核算

## 技术架构

### 整体架构
```
用户请求 → API网关 → Agent调度器 → Kubernetes集群 → Agent容器
       ↓          ↓          ↓                ↓
   认证授权     任务队列     资源评估     容器运行时
       ↓          ↓          ↓                ↓
   监控日志     策略引擎     调度决策     状态反馈
```

### 组件设计
1. **API服务层**：FastAPI提供RESTful API，处理用户请求
2. **调度引擎**：基于资源可用性和策略的智能调度器
3. **容器运行时**：通过Kubernetes API管理Docker容器
4. **资源管理器**：监控和分配集群资源（CPU/内存/GPU）
5. **监控系统**：收集容器指标和运行状态
6. **存储服务**：Redis缓存 + PostgreSQL数据库

## 实现方案

### 1. API服务实现
```python
# Agent管理API
@app.post("/api/v1/agents")
async def create_agent(agent_spec: AgentSpec):
    """创建新的Agent实例"""
    # 1. 验证资源配额
    if not await resource_manager.check_quota(agent_spec):
        raise HTTPException(status_code=400, detail="资源配额不足")

    # 2. 调度决策
    node = await scheduler.schedule_agent(agent_spec)

    # 3. 创建容器
    agent_id = await container_manager.create_agent(
        agent_spec, node
    )

    # 4. 记录状态
    await db_manager.create_agent_record(agent_id, agent_spec)

    return {"agent_id": agent_id, "status": "creating"}

@app.get("/api/v1/agents/{agent_id}")
async def get_agent_status(agent_id: str):
    """获取Agent状态"""
    status = await container_manager.get_agent_status(agent_id)
    metrics = await monitor.get_agent_metrics(agent_id)

    return {
        "agent_id": agent_id,
        "status": status,
        "metrics": metrics
    }

@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """删除Agent实例"""
    await container_manager.delete_agent(agent_id)
    await db_manager.delete_agent_record(agent_id)

    return {"status": "deleted"}
```

### 2. 调度器实现
```python
class AgentScheduler:
    def __init__(self):
        self.nodes = {}  # 节点资源信息
        self.policies = {
            "spread": self._spread_policy,
            "pack": self._pack_policy,
            "balanced": self._balanced_policy
        }

    async def schedule_agent(self, agent_spec: AgentSpec) -> str:
        """调度Agent到合适的节点"""
        # 过滤可用节点
        available_nodes = await self._filter_available_nodes(agent_spec)

        if not available_nodes:
            raise NoAvailableNodeError("没有可用节点")

        # 应用调度策略
        policy = agent_spec.scheduling_policy or "balanced"
        selected_node = await self.policies[policy](available_nodes, agent_spec)

        # 更新节点资源
        await self._allocate_resources(selected_node, agent_spec)

        return selected_node

    async def _spread_policy(self, nodes, agent_spec):
        """分散策略：尽可能将Agent分散到不同节点"""
        # 按负载升序排序
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[1]["load"])
        return sorted_nodes[0][0]  # 选择负载最低的节点

    async def _pack_policy(self, nodes, agent_spec):
        """紧凑策略：尽可能集中到少数节点"""
        # 按剩余资源降序排序
        sorted_nodes = sorted(
            nodes.items(),
            key=lambda x: x[1]["available_cpu"],
            reverse=True
        )
        return sorted_nodes[0][0]  # 选择剩余资源最多的节点

    async def _balanced_policy(self, nodes, agent_spec):
        """平衡策略：综合考虑CPU、内存、网络负载"""
        scored_nodes = []
        for node_id, node_info in nodes.items():
            score = self._calculate_score(node_info, agent_spec)
            scored_nodes.append((score, node_id))

        # 选择分数最高的节点
        scored_nodes.sort(reverse=True)
        return scored_nodes[0][1]

    def _calculate_score(self, node_info, agent_spec):
        """计算节点综合评分"""
        cpu_score = node_info["available_cpu"] / agent_spec.cpu_request
        mem_score = node_info["available_memory"] / agent_spec.memory_request

        # 考虑节点负载因子
        load_factor = 1.0 - node_info["load"]

        # 综合评分（可调整权重）
        total_score = (cpu_score * 0.4 + mem_score * 0.4 + load_factor * 0.2)
        return total_score
```

### 3. 容器管理器实现
```python
class ContainerManager:
    def __init__(self, k8s_client):
        self.k8s_client = k8s_client
        self.namespace = "agent-platform"

    async def create_agent(self, agent_spec: AgentSpec, node: str) -> str:
        """创建Agent容器"""
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"

        # 构建Pod配置
        pod_spec = self._build_pod_spec(agent_id, agent_spec, node)

        # 创建Kubernetes Pod
        await self.k8s_client.create_namespaced_pod(
            namespace=self.namespace,
            body=pod_spec
        )

        # 创建Service（如果需要网络访问）
        if agent_spec.expose_service:
            service_spec = self._build_service_spec(agent_id)
            await self.k8s_client.create_namespaced_service(
                namespace=self.namespace,
                body=service_spec
            )

        return agent_id

    def _build_pod_spec(self, agent_id: str, agent_spec: AgentSpec, node: str):
        """构建Pod配置"""
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": agent_id,
                "labels": {
                    "app": "agent",
                    "agent-id": agent_id,
                    "owner": agent_spec.owner
                }
            },
            "spec": {
                "nodeName": node,  # 指定调度节点
                "restartPolicy": "Never",
                "containers": [{
                    "name": "agent-container",
                    "image": agent_spec.image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": agent_spec.command,
                    "args": agent_spec.args,
                    "resources": {
                        "requests": {
                            "cpu": f"{agent_spec.cpu_request}m",
                            "memory": f"{agent_spec.memory_request}Mi"
                        },
                        "limits": {
                            "cpu": f"{agent_spec.cpu_limit}m",
                            "memory": f"{agent_spec.memory_limit}Mi",
                            "nvidia.com/gpu": str(agent_spec.gpu_count) if agent_spec.gpu_count > 0 else None
                        }
                    },
                    "env": [
                        {"name": "AGENT_ID", "value": agent_id},
                        {"name": "REDIS_HOST", "value": "redis-service"}
                    ],
                    "securityContext": {
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True
                    },
                    "volumeMounts": [
                        {
                            "name": "agent-data",
                            "mountPath": "/data",
                            "readOnly": False
                        }
                    ]
                }],
                "volumes": [
                    {
                        "name": "agent-data",
                        "emptyDir": {}
                    }
                ]
            }
        }

    async def get_agent_status(self, agent_id: str) -> dict:
        """获取Agent状态"""
        try:
            pod = await self.k8s_client.read_namespaced_pod(
                name=agent_id,
                namespace=self.namespace
            )

            return {
                "phase": pod.status.phase,
                "start_time": pod.status.start_time,
                "container_statuses": [
                    {
                        "name": cs.name,
                        "state": str(cs.state),
                        "ready": cs.ready
                    }
                    for cs in pod.status.container_statuses or []
                ]
            }
        except Exception as e:
            return {"phase": "unknown", "error": str(e)}
```

### 4. 资源隔离配置
```yaml
# Kubernetes Security Context配置
securityContext:
  # 非root用户运行
  runAsUser: 1000
  runAsGroup: 1000

  # 权限控制
  capabilities:
    drop: ["ALL"]
  allowPrivilegeEscalation: false

  # 文件系统保护
  readOnlyRootFilesystem: true
  seccompProfile:
    type: "RuntimeDefault"

  # AppArmor配置（可选）
  apparmorProfile: "runtime/default"

# 网络策略
networkPolicy:
  podSelector:
    matchLabels:
      app: agent
  policyTypes:
  - Ingress
  - Egress
  ingress: []  # 默认拒绝所有入站
  egress: []   # 默认拒绝所有出站
```

## 项目结构

```
agent-platform/
├── api-service/              # API服务
│   ├── app/
│   │   ├── api/             # API路由
│   │   ├── core/            # 核心配置
│   │   ├── models/          # 数据模型
│   │   ├── services/        # 业务服务
│   │   └── utils/           # 工具函数
│   ├── alembic/             # 数据库迁移
│   ├── tests/               # 测试文件
│   └── Dockerfile           # 容器配置
├── scheduler/                # 调度引擎
│   ├── src/
│   │   ├── policies/        # 调度策略
│   │   ├── evaluators/      # 资源评估器
│   │   └── models/          # 调度模型
│   └── Dockerfile
├── monitor/                  # 监控系统
│   ├── collectors/          # 指标收集器
│   ├── exporters/           # 指标导出
│   └── alerts/              # 告警规则
├── deployments/              # 部署配置
│   ├── kubernetes/          # K8s资源定义
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── networkpolicy.yaml
│   └── docker-compose.yml    # 本地开发
├── agent-samples/            # Agent示例
│   ├── simple-python/       # Python Agent示例
│   ├── langchain-agent/     # LangChain Agent示例
│   └── custom-image/        # 自定义镜像示例
├── docs/                     # 项目文档
└── Makefile                  # 构建脚本
```

## 开发计划

### 第1天：基础框架搭建
1. **上午**：API服务基础架构
   - 搭建FastAPI项目框架
   - 设计数据模型和API接口
   - 集成数据库和缓存

2. **下午**：Kubernetes集成
   - 配置Kubernetes客户端
   - 实现容器管理基础功能
   - 测试Pod创建和删除

3. **晚上**：监控和日志
   - 集成Prometheus指标
   - 配置结构化日志
   - 编写基础测试

### 第2天：核心功能实现
1. **上午**：调度器实现
   - 实现资源监控和发现
   - 开发调度策略算法
   - 测试调度决策逻辑

2. **下午**：安全与隔离
   - 配置安全上下文
   - 实现网络策略
   - 测试资源限制和隔离

3. **晚上**：集成与优化
   - 组件集成测试
   - 性能优化调整
   - 编写用户文档

## 测试策略

### 单元测试
```python
def test_scheduler_policies():
    """测试调度策略"""
    scheduler = AgentScheduler()

    # 模拟节点数据
    nodes = {
        "node1": {"available_cpu": 2000, "available_memory": 4096, "load": 0.3},
        "node2": {"available_cpu": 1000, "available_memory": 2048, "load": 0.7}
    }

    # 测试分散策略
    agent_spec = AgentSpec(cpu_request=500, memory_request=1024)
    agent_spec.scheduling_policy = "spread"

    selected = scheduler._spread_policy(nodes, agent_spec)
    assert selected == "node1"  # 应该选择负载低的节点

    # 测试紧凑策略
    agent_spec.scheduling_policy = "pack"
    selected = scheduler._pack_policy(nodes, agent_spec)
    assert selected == "node1"  # 应该选择资源多的节点
```

### 集成测试
```python
async def test_agent_lifecycle():
    """测试Agent完整生命周期"""
    # 创建Agent
    response = await client.post("/api/v1/agents", json={
        "image": "python:3.11-slim",
        "command": ["python", "-c", "print('Hello Agent')"],
        "cpu_request": 100,
        "memory_request": 128
    })
    assert response.status_code == 200
    agent_id = response.json()["agent_id"]

    # 检查状态
    response = await client.get(f"/api/v1/agents/{agent_id}")
    assert response.status_code == 200
    status = response.json()["status"]
    assert status["phase"] in ["Pending", "Running", "Succeeded"]

    # 删除Agent
    response = await client.delete(f"/api/v1/agents/{agent_id}")
    assert response.status_code == 200
```

### 安全测试
```python
def test_security_isolation():
    """测试安全隔离"""
    # 测试非root用户运行
    pod_spec = container_manager._build_pod_spec(...)
    security_context = pod_spec["spec"]["containers"][0]["securityContext"]

    assert security_context["runAsUser"] == 1000
    assert security_context["runAsGroup"] == 1000
    assert security_context["allowPrivilegeEscalation"] == False
    assert security_context["readOnlyRootFilesystem"] == True
```

## 部署方案

### 开发环境（Docker Compose）
```yaml
version: '3.8'
services:
  api-service:
    build: ./api-service
    ports:
      - "8000:8000"
    environment:
      - KUBECONFIG=/kube/config
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/agentdb
    volumes:
      - ./kubeconfig:/kube/config:ro

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=agentdb
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

  minikube:  # 本地Kubernetes集群
    image: minikube/minikube
    command: start
    privileged: true
    volumes:
      - /lib/modules:/lib/modules
      - /var/run/docker.sock:/var/run/docker.sock

volumes:
  postgres_data:
  redis_data:
```

### 生产环境（Kubernetes）
```yaml
# API服务部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  namespace: agent-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  template:
    metadata:
      labels:
        app: agent-api
    spec:
      containers:
      - name: api
        image: agent-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

# 服务暴露
apiVersion: v1
kind: Service
metadata:
  name: agent-api-service
  namespace: agent-platform
spec:
  selector:
    app: agent-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 学习收获

### 技术技能
1. ✅ 掌握Kubernetes API编程和容器管理
2. ✅ 理解容器调度算法和资源管理
3. ✅ 学会构建安全的容器运行时环境
4. ✅ 掌握微服务架构下的Agent平台设计

### 工程能力
1. ✅ 云原生应用开发能力
2. ✅ 多组件系统集成经验
3. ✅ 安全隔离和资源控制能力
4. ✅ 可观测性和监控体系建设

### 架构思维
1. ✅ 分布式系统设计能力
2. ✅ 弹性伸缩和容错设计
3. ✅ 多租户和配额管理设计
4. ✅ 平台化思维和API设计

## 扩展方向

### 功能扩展
1. **工作流引擎**：支持多步骤Agent工作流
2. **模型管理**：大模型版本管理和部署
3. **联邦学习**：跨集群Agent协作
4. **成本优化**：智能资源分配和成本控制

### 架构优化
1. **边缘计算**：支持边缘节点部署
2. **Serverless架构**：按需启动Agent实例
3. **多集群管理**：跨云、跨地域集群统一管理
4. **AI调度**：基于预测的智能调度

### 生态集成
1. **LangChain集成**：支持LangChain Agent直接部署
2. **向量数据库**：集成向量存储和检索
3. **监控告警**：与现有监控系统集成
4. **CI/CD流水线**：Agent自动构建和部署

---

*项目特点：聚焦Agent运行时核心问题，实践云原生和容器安全技术*
*学习价值：深入理解AI基础设施设计和实现，为实际生产环境打下基础*

## 相关学习路径
- [Python现代化开发](./../learning-paths/02-python-modern.md)
- [云原生进阶](./../learning-paths/06-cloud-native-advanced.md)
- [分布式系统复习](./../learning-paths/05-distributed-systems-review.md)