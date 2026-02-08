# 实践项目：高性能API网关

## 项目概述
- **目标**：实现一个基于Go和FastAPI的混合架构API网关
- **技术栈**：Go + FastAPI + Redis + Prometheus
- **时间**：春节第1周后2天完成
- **关联学习**：Go复习 + Python现代化开发

## 项目背景
现代微服务架构中，API网关作为系统的入口，承担着路由、认证、限流、监控等重要功能。本项目旨在结合Go的高性能和Python的快速开发优势，构建一个实用的API网关。

## 功能需求

### 核心功能
1. **请求路由**：基于路径和方法的请求转发
2. **负载均衡**：支持轮询、权重等负载均衡策略
3. **限流保护**：基于IP、用户、接口的限流控制
4. **认证授权**：JWT验证、API密钥认证
5. **监控指标**：请求统计、性能指标、错误率

### 高级功能（可选）
1. **熔断降级**：服务故障时的熔断保护
2. **缓存加速**：响应缓存减少后端压力
3. **日志收集**：结构化日志记录
4. **配置热更新**：动态配置无需重启

## 技术架构

### 整体架构
```
客户端请求 → API网关 → 后端服务
       ↓          ↓          ↓
   限流认证     路由转发     业务处理
       ↓          ↓          ↓
   监控日志     负载均衡     数据返回
```

### 组件设计
1. **Go核心引擎**：高性能请求处理、限流算法、连接池管理
2. **FastAPI管理接口**：配置管理、监控面板、动态路由
3. **Redis存储**：限流计数、会话缓存、配置缓存
4. **Prometheus监控**：指标收集、告警触发

## 实现方案

### 1. Go核心组件实现
```go
// 网关核心结构
type APIGateway struct {
    router      *Router          // 路由管理器
    rateLimiter *RateLimiter     // 限流器
    authManager *AuthManager     // 认证管理器
    metrics     *MetricsCollector // 指标收集器
}

// 请求处理流程
func (g *APIGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // 1. 认证验证
    if !g.authManager.Authenticate(r) {
        w.WriteHeader(http.StatusUnauthorized)
        return
    }

    // 2. 限流检查
    if !g.rateLimiter.Allow(r) {
        w.WriteHeader(http.StatusTooManyRequests)
        return
    }

    // 3. 路由匹配
    backend := g.router.Match(r)

    // 4. 转发请求
    g.forwardRequest(w, r, backend)

    // 5. 记录指标
    g.metrics.RecordRequest(r, backend)
}
```

### 2. FastAPI管理接口
```python
# 管理API路由
@app.get("/api/routes")
async def list_routes():
    """列出所有路由配置"""
    return await route_service.get_all_routes()

@app.post("/api/routes")
async def create_route(route: RouteSchema):
    """创建新路由"""
    return await route_service.create_route(route)

@app.get("/api/metrics")
async def get_metrics():
    """获取网关指标"""
    return await metrics_service.get_metrics()
```

### 3. 限流器实现（令牌桶算法）
```go
// 令牌桶限流器
type TokenBucketLimiter struct {
    capacity    int           // 桶容量
    tokens      float64       // 当前令牌数
    fillRate    float64       // 填充速率（令牌/秒）
    lastRefill  time.Time     // 上次填充时间
    mu          sync.Mutex    // 并发锁
}

func (t *TokenBucketLimiter) Allow() bool {
    t.mu.Lock()
    defer t.mu.Unlock()

    // 计算需要填充的令牌
    now := time.Now()
    elapsed := now.Sub(t.lastRefill).Seconds()
    t.tokens = math.Min(float64(t.capacity), t.tokens + elapsed * t.fillRate)
    t.lastRefill = now

    // 检查是否有足够令牌
    if t.tokens >= 1 {
        t.tokens -= 1
        return true
    }
    return false
}
```

## 项目结构

```
api-gateway/
├── go-core/                    # Go核心组件
│   ├── cmd/gateway/           # 主程序入口
│   ├── internal/
│   │   ├── router/            # 路由组件
│   │   ├── middleware/        # 中间件组件
│   │   ├── rate_limiter/      # 限流器
│   │   ├── auth/              # 认证组件
│   │   └── metrics/           # 指标收集
│   ├── pkg/
│   │   ├── config/            # 配置管理
│   │   └── utils/             # 工具函数
│   └── Dockerfile             # Go容器配置
├── python-admin/               # Python管理接口
│   ├── app/
│   │   ├── api/               # API路由
│   │   ├── core/              # 核心配置
│   │   ├── services/          # 业务服务
│   │   └── models/            # 数据模型
│   ├── alembic/               # 数据库迁移
│   └── Dockerfile             # Python容器配置
├── deployments/                # 部署配置
│   ├── docker-compose.yml     # 本地开发
│   ├── kubernetes/            # K8s部署
│   └── config/                # 配置文件
├── tests/                      # 测试文件
├── docs/                       # 项目文档
└── Makefile                    # 构建脚本
```

## 开发计划

### 第1天：基础框架搭建
1. **上午**：Go核心组件基础架构
   - 实现HTTP服务器框架
   - 设计路由匹配算法
   - 创建基本中间件系统

2. **下午**：核心功能实现
   - 实现令牌桶限流器
   - 开发JWT认证中间件
   - 集成Prometheus指标

3. **晚上**：测试验证
   - 编写单元测试
   - 性能基准测试
   - 文档编写

### 第2天：管理接口与集成
1. **上午**：FastAPI管理接口
   - 创建管理API服务
   - 实现路由配置管理
   - 开发监控面板

2. **下午**：系统集成与优化
   - Go与Python服务通信
   - Redis缓存集成
   - 性能优化调整

3. **晚上**：部署与测试
   - Docker容器化
   - 集成测试
   - 项目总结

## 测试策略

### 单元测试
```go
func TestRateLimiter(t *testing.T) {
    limiter := NewTokenBucketLimiter(10, 1) // 容量10，速率1/s

    // 测试限流逻辑
    allowed := 0
    for i := 0; i < 15; i++ {
        if limiter.Allow() {
            allowed++
        }
    }

    assert.Equal(t, 10, allowed, "应该只允许10次请求")
}
```

### 集成测试
```python
async def test_api_gateway():
    # 测试完整请求流程
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 测试正常请求
        response = await ac.get("/api/users")
        assert response.status_code == 200

        # 测试限流
        for _ in range(20):
            response = await ac.get("/api/users")
        assert response.status_code == 429  # 太多请求
```

### 性能测试
```bash
# 使用wrk进行压力测试
wrk -t12 -c400 -d30s http://localhost:8080/api/test

# 使用ab进行并发测试
ab -n 10000 -c 100 http://localhost:8080/api/test
```

## 部署方案

### 开发环境（Docker Compose）
```yaml
version: '3.8'
services:
  gateway:
    build: ./go-core
    ports:
      - "8080:8080"
    depends_on:
      - redis

  admin:
    build: ./python-admin
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./deployments/config/prometheus.yml:/etc/prometheus/prometheus.yml
```

### 生产环境（Kubernetes）
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: gateway
        image: api-gateway:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## 学习收获

### 技术技能
1. ✅ 掌握Go高性能网络编程
2. ✅ 理解限流算法和实现
3. ✅ 学会混合架构设计
4. ✅ 掌握API网关核心功能

### 工程能力
1. ✅ 微服务网关设计能力
2. ✅ 多语言系统集成经验
3. ✅ 性能优化和测试能力
4. ✅ 容器化部署经验

## 扩展方向

### 功能扩展
1. **服务发现集成**：集成Consul/etcd自动发现
2. **协议支持**：支持gRPC、WebSocket等协议
3. **安全增强**：WAF功能、DDoS防护
4. **流量镜像**：生产流量复制到测试环境

### 架构优化
1. **水平扩展**：无状态设计支持水平扩展
2. **高可用**：多活部署、故障自动转移
3. **性能优化**：连接池优化、内存管理
4. **可观测性**：更丰富的监控指标和告警

---

*项目特点：结合Go性能优势和Python开发效率的混合架构*
*学习价值：深入理解API网关核心原理和实现*