# 系统安全基础（2天）

## 概述
- **目标**：系统掌握系统安全的基础知识与实践技能，理解常见安全威胁与防护措施，满足JD中"构建高性能、高安全性的Agent运行时环境"的要求，为构建安全可靠的系统打下基础
- **时间**：春节第3周（2天）
- **前提**：了解基本计算机网络概念，有编程经验
- **强度**：中等强度（每天6-8小时），适合需要了解安全基础的工程师

## JD要求对应

### JD领域覆盖
| JD领域 | 对应内容 | 优先级 |
|--------|----------|--------|
| 一、高并发服务端与API系统 | API安全、认证授权、输入验证 | ⭐⭐⭐ |
| 二、大规模数据处理Pipeline | 数据加密、访问控制、审计日志 | ⭐⭐ |
| 三、Agent基础设施与运行时平台 | 容器安全、隔离机制、最小权限 | ⭐⭐⭐ |
| 四、异构超算基础设施 | 硬件安全、密钥管理、安全通信 | ⭐⭐ |

### JD能力对应
| 能力要求 | 学习内容 | 验证方式 |
|----------|----------|----------|
| **系统安全意识** | 安全威胁、漏洞类型、攻击手法 | 安全评估 |
| **安全防护能力** | 加密技术、认证授权、输入验证 | 安全编码 |
| **运维安全** | 监控告警、应急响应、合规审计 | 安全运维 |

## 学习重点

### 1. 安全基础概念（第1天上午）
**JD引用**："构建高性能、高安全性的Agent运行时环境"

**核心内容**：
- 安全三要素（CIA）
  - **机密性（Confidentiality）**：防止未授权访问
  - **完整性（Integrity）**：防止未授权修改
  - **可用性（Availability）**：确保授权访问
- 常见安全威胁
  - **DDoS攻击**：分布式拒绝服务
  - **SQL注入**：数据库攻击
  - **XSS攻击**：跨站脚本
  - **CSRF攻击**：跨站请求伪造
  - **中间人攻击（MITM）**：通信窃听
  - **暴力破解**：密码攻击
  - **社会工程**：人为因素
- 安全防护原则
  - **最小权限原则**：只授予必要权限
  - **纵深防御**：多层安全防护
  - **零信任**：不信任内部网络
  - **默认安全**：安全默认值
  - **开放设计**：不依赖安全性保密

**实践任务**：
- 分析常见漏洞案例
- 理解安全三要素
- 制定安全策略

### 2. 密码学基础（第1天下午）
**JD引用**："深刻理解计算机组成、操作系统、计算机网络等核心原理"

**核心内容**：
- 对称加密
  - **AES（Advanced Encryption Standard）**
    - 128/192/256位密钥
    - ECB/CBC/GCM模式
    - 分组加密原理
  - **DES/3DES**：已淘汰算法
- 非对称加密
  - **RSA**
    - 大数分解原理
    - 密钥生成过程
    - 应用场景
  - **ECC（椭圆曲线密码学）**
    - ECDH密钥交换
    - ECDSA数字签名
    - 安全性优势
- 哈希函数
  - **SHA-256/SHA-3**
    - 单向哈希
    - 碰撞抵抗
    - 应用场景
  - **bcrypt/scrypt/Argon2**
    - 密码哈希
    - 加盐处理
    - 工作因子
- 数字证书与PKI
  - **X.509证书**
    - 证书结构
    - 证书链
    - 证书验证
  - **TLS/SSL协议**
    - 握手过程
    - 证书验证
    - 加密套件

**实践任务**：
- 使用OpenSSL生成证书
- 实现AES加密解密
- 实现密码哈希
- 分析TLS握手过程

### 3. 认证与授权（第1天晚上）
**JD引用**："对Kubernetes及云原生部署有深入理解，具备云上系统优化经验"

**核心内容**：
- 认证机制
  - **用户名密码认证**
    - 密码策略
    - 密码存储（加盐哈希）
    - 登录限制
  - **多因素认证（MFA）**
    - TOTP动态口令
    - SMS/邮件验证码
    - 硬件令牌（YubiKey）
  - **OAuth 2.0**
    - 授权流程
    - 令牌类型（Access/Refresh）
    - 权限范围（Scope）
  - **OpenID Connect**
    - ID Token
    - 用户信息端点
- 授权模型
  - **RBAC（基于角色的访问控制）**
    - 用户-角色-权限
    - 角色继承
    - 最小权限
  - **ABAC（基于属性的访问控制）**
    - 属性匹配
    - 动态策略
    - 细粒度控制
  - **ACL（访问控制列表）**
    - 直接权限
    - 文件权限
- JWT（JSON Web Token）
  - **JWT结构**：Header、Payload、Signature
  - **JWT签名**：HS256、RS256、ES256
  - **JWT验证**：过期时间、发行者验证
  - **JWT刷新**：Refresh Token机制

**实践任务**：
- 实现OAuth 2.0认证
- 实现JWT认证
- 设计RBAC权限系统
- 配置MFA认证

### 4. 应用安全（第2天上午）
**JD引用**："熟练运用Profiling和可观测性工具分析与定位复杂系统问题"

**核心内容**：
- OWASP Top 10
  - **注入攻击（Injection）**
    - SQL注入原理
    - 防御方法（参数化查询）
    - NoSQL注入
    - 命令注入
  - **身份认证缺陷（Broken Authentication）**
    - 弱密码策略
    - 会话管理漏洞
    - 凭证填充
  - **敏感数据泄露**
    - 加密存储
    - 传输加密
    - 日志脱敏
  - **XML外部实体（XXE）**
    - XXE原理
    - 防御方法
  - **访问控制失效**
    - 水平越权
    - 垂直越权
    - IDOR漏洞
- 安全编码实践
  - **输入验证**
    - 白名单验证
    - 类型检查
    - 长度限制
  - **输出编码**
    - HTML编码
    - URL编码
    - JSON编码
  - **安全配置**
    - 安全的HTTP头
    - CORS配置
    - CSP策略

**实践任务**：
- 实现SQL注入防御
- 配置安全HTTP头
- 实现输入验证
- 进行安全代码审计

### 5. 网络安全（第2天下午）
**JD引用**："面向数千万日活用户的产品后端架构设计"

**核心内容**：
- TLS/SSL深入
  - **TLS 1.3特性**
    - 1-RTT握手
    - 前向安全性
    - 0-RTT恢复
  - **证书管理**
    - Let's Encrypt证书
    - 证书自动化（Cert-manager）
    - 证书透明度日志
  - **加密套件**
    - 强密码套件
    - 禁用弱算法
    - OCSP Stapling
- DDoS防护
  - **攻击类型**
    - 流量型（SYN Flood、UDP Flood）
    - 协议型（Slowloris）
    - 应用型（HTTP Flood）
  - **防护措施**
    - 流量清洗
    - CDN加速
    - 速率限制
    - Anycast分发
- Web应用防火墙（WAF）
  - **规则类型**
    - 基础规则（OWASP）
    - 自定义规则
    - 机器学习检测
  - **部署模式**
    - 反向代理
    - 云WAF
    - 主机WAF

**实践任务**：
- 配置TLS 1.3
- 部署WAF规则
- 配置Rate Limiting
- 分析TLS配置

### 6. 容器与云原生安全（第2天下午）
**JD引用**："设计与开发支撑海量AI Agent运行的下一代容器调度与隔离平台"

**核心内容**：
- 容器安全
  - **镜像安全**
    - 最小基础镜像
    - 镜像扫描（Trivy、Clair）
    - 镜像签名（Notary）
    - 私有仓库安全
  - **运行时安全**
    - 只读文件系统
    - 能力限制（Capabilities）
    - Seccomp/AppArmor
    - 进程权限
  - **容器隔离**
    - 命名空间隔离
    - Cgroup资源限制
    - 网络隔离
- Kubernetes安全
  - **RBAC配置**
    - ServiceAccount
    - 角色绑定
    - 最小权限
  - **网络策略**
    - NetworkPolicy
    - 命名空间隔离
    - 出口规则
  - **密钥管理**
    - Kubernetes Secrets
    - HashiCorp Vault
    - 密钥轮换
- 服务网格安全
  - **mTLS认证**
    - 服务间认证
    - 自动证书发行
  - **授权策略**
    - 细粒度访问控制
    - 认证与授权分离

**实践任务**：
- 配置K8s RBAC
- 部署容器镜像扫描
- 配置NetworkPolicy
- 启用服务网格mTLS

### 7. 安全监控与应急响应（第2天晚上）
**JD引用**："熟练运用Profiling和可观测性工具分析与定位复杂系统问题"

**核心内容**：
- 安全监控
  - **日志收集**
    - 认证日志
    - 访问日志
    - 审计日志
  - **入侵检测**
    - IDS/IPS
    - HIDS（主机入侵检测）
    - NIDS（网络入侵检测）
  - **异常检测**
    - 行为分析
    - 机器学习检测
    - 威胁情报
- 应急响应
  - **响应流程**
    - 准备 → 检测 → 遏制 → 根除 → 恢复 → 复盘
  - **事件分类**
    - P0（严重）
    - P1（高）
    - P2（中）
    - P3（低）
  - **证据保全**
    - 日志保存
    - 内存转储
    - 磁盘镜像
- 合规审计
  - **审计标准**
    - SOC 2
    - ISO 27001
    - GDPR
    - 等保2.0
  - **审计工具**
    - OpenSCAP
    - Falco
    - Auditd

**实践任务**：
- 配置安全日志收集
- 部署入侵检测系统
- 制定应急响应流程
- 进行安全审计

## 实践项目：安全API服务

### 项目目标
**JD对应**：满足"构建高性能、高安全性的Agent运行时环境"要求

实现一个安全的API服务，包含：
1. JWT认证与授权
2. 密码安全存储
3. 输入验证与防护
4. TLS加密传输
5. 安全监控与告警

### 技术栈参考（明确版本）
- **认证**：JWT（jsonwebtoken 9.0+）
- **密码哈希**：bcrypt 4.1+ / Argon2
- **输入验证**：Zod 3.21+ / Joi 17.9+
- **安全头**：Helmet 7.0+ / CORS
- **限流**：express-rate-limit 7.0+ / Redis
- **日志**：Winston 3.11+ / ELK Stack

### 环境配置要求
- **操作系统**：Linux（推荐Ubuntu 22.04）
- **依赖**：
  ```bash
  # Node.js安全依赖
  npm install jsonwebtoken bcrypt helmet joi express-rate-limit winston

  # 安全扫描工具
  npm install --save-dev nps eslint-plugin-security

  # Docker部署
  docker-compose -f security.yml up -d
  ```

### 架构设计
```
secure-api/
├── src/
│   ├── auth/                  # 认证模块
│   │   ├── jwt.service.ts    # JWT服务
│   │   ├── oauth.service.ts   # OAuth服务
│   │   └── mfa.service.ts    # MFA服务
│   ├── security/              # 安全模块
│   │   ├── encryption.service.ts  # 加密服务
│   │   ├── validation.service.ts # 输入验证
│   │   └── rate-limit.service.ts # 限流服务
│   ├── middleware/            # 中间件
│   │   ├── auth.middleware.ts    # 认证中间件
│   │   ├── validation.middleware.ts  # 验证中间件
│   │   ├── rate-limit.middleware.ts # 限流中间件
│   │   └── security-headers.middleware.ts  # 安全头中间件
│   ├── monitoring/            # 监控模块
│   │   ├── logger.service.ts  # 日志服务
│   │   ├── alert.service.ts   # 告警服务
│   │   └── audit.service.ts   # 审计服务
│   └── config/                # 配置
│       ├── security.config.ts  # 安全配置
│       └── tls.config.ts      # TLS配置
├── tests/
│   ├── security/             # 安全测试
│   │   ├── injection.test.ts  # 注入测试
│   │   ├── auth.test.ts      # 认证测试
│   │   └── headers.test.ts   # 安全头测试
│   └── penetration/          # 渗透测试
│       └── owasp.test.ts     # OWASP测试
├── scripts/
│   ├── security-scan.sh      # 安全扫描脚本
│   └── tls-setup.sh          # TLS配置脚本
└── docker/
    ├── Dockerfile
    └── security.yml
```

### 核心组件设计

#### 1. JWT认证
```typescript
// auth/jwt.service.ts
import jwt from 'jsonwebtoken';
import { Injectable } from '@nestjs/common';

@Injectable()
export class JwtService {
  private readonly secret: string;
  private readonly refreshSecret: string;
  private readonly accessTokenExpires = '15m';
  private readonly refreshTokenExpires = '7d';

  constructor() {
    this.secret = process.env.JWT_SECRET!;
    this.refreshSecret = process.env.JWT_REFRESH_SECRET!;
  }

  // 生成Access Token
  generateAccessToken(userId: string, role: string): string {
    return jwt.sign(
      {
        sub: userId,
        role,
        type: 'access',
      },
      this.secret,
      { expiresIn: this.accessTokenExpires }
    );
  }

  // 生成Refresh Token
  generateRefreshToken(userId: string): string {
    return jwt.sign(
      {
        sub: userId,
        type: 'refresh',
      },
      this.refreshSecret,
      { expiresIn: this.refreshTokenExpires }
    );
  }

  // 验证Token
  verifyToken(token: string, isRefresh: boolean = false): JwtPayload {
    const secret = isRefresh ? this.refreshSecret : this.secret;

    try {
      return jwt.verify(token, secret) as JwtPayload;
    } catch (error) {
      throw new UnauthorizedException('Invalid token');
    }
  }

  // Token黑名单
  async isBlacklisted(token: string): Promise<boolean> {
    const redis = getRedisClient();
    const blacklisted = await redis.get(`blacklist:${token}`);
    return !!blacklisted;
  }
}
```

#### 2. 输入验证
```typescript
// security/validation.service.ts
import Joi from 'joi';

export const UserRegistrationSchema = Joi.object({
  email: Joi.string()
    .email({ tlds: { allow: false } })  // 验证邮箱格式
    .max(255)
    .required(),

  password: Joi.string()
    .min(12)                              // 最小12位
    .max(128)
    .regex(/^(?=.*[a-z])/, 'lowercase')  // 小写字母
    .regex(/^(?=.*[A-Z])/, 'uppercase')  // 大写字母
    .regex(/^(?=.*[0-9])/, 'number')      // 数字
    .regex(/^(?=.*[!@#$%^&*])/, 'special') // 特殊字符
    .required()
    .messages({
      'string.pattern.base': 'Password does not meet complexity requirements',
    }),

  username: Joi.string()
    .alphanum()                           // 只允许字母数字
    .min(3)
    .max(30)
    .required(),
});

export const sanitizeInput = (input: string): string => {
  // 移除HTML标签
  let sanitized = input.replace(/<[^>]*>/g, '');

  // 移除SQL注入特征
  sanitized = sanitized.replace(/(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|EXEC)\b)/gi, '');

  // 移除多余空白
  sanitized = sanitized.trim();

  return sanitized;
};
```

#### 3. 安全头中间件
```typescript
// middleware/security-headers.middleware.ts
import { Request, Response, NextFunction } from 'express';

export const securityHeaders = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  // 防止XSS攻击
  res.setHeader('X-XSS-Protection', '1; mode=block');

  // 防止点击劫持
  res.setHeader('X-Frame-Options', 'DENY');

  // 防止MIME类型嗅探
  res.setHeader('X-Content-Type-Options', 'nosniff');

  // 启用严格传输安全（HSTS）
  res.setHeader('Strict-Transport-Security',
    'max-age=31536000; includeSubDomains; preload');

  // 引用策略
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

  // 权限策略
  res.setHeader('Permissions-Policy',
    'geolocation=(), microphone=(), camera=()');

  // CSP策略
  res.setHeader('Content-Security-Policy',
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'");

  next();
};
```

#### 4. 限流
```typescript
// security/rate-limit.service.ts
import rateLimit from 'express-rate-limit';
import Redis from 'ioredis';
import { Request, Response } from 'express';

const redis = new Redis(process.env.REDIS_URL!);

// 创建IP限流器
export const createRateLimiter = (options: {
  windowMs: number;
  max: number;
  message: string;
}) => {
  return rateLimit({
    store: new RedisStore({
      client: redis,
      prefix: 'rl:',
    }),
    windowMs: options.windowMs,
    max: options.max,
    message: {
      error: options.message,
      retryAfter: Math.ceil(options.windowMs / 1000),
    },
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req: Request) => {
      // 按IP限流，考虑X-Forwarded-For
      const forwarded = req.headers['x-forwarded-for'];
      if (typeof forwarded === 'string') {
        return forwarded.split(',')[0].trim();
      }
      return req.ip || req.socket.remoteAddress!;
    },
    skip: (req: Request) => {
      // 白名单跳过
      const whitelist = process.env.RATE_LIMIT_WHITELIST?.split(',') || [];
      return whitelist.includes(req.ip!);
    },
    handler: (req: Request, res: Response) => {
      // 记录限流日志
      logger.warn('Rate limit exceeded', {
        ip: req.ip,
        path: req.path,
        method: req.method,
      });
      res.status(429).json({
        error: 'Too Many Requests',
        message: options.message,
      });
    },
  });
};

// 认证限流（更严格）
export const authRateLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000,  // 15分钟
  max: 5,                      // 最多5次尝试
  message: 'Too many login attempts, please try again later',
});

// API限流（标准）
export const apiRateLimiter = createRateLimiter({
  windowMs: 60 * 1000,        // 1分钟
  max: 100,                    // 最多100次请求
  message: 'API rate limit exceeded',
});
```

## 学习资源

### 经典书籍
1. **《Web应用安全权威指南》**：OWASP官方指南
2. **《黑客与画家》**：安全思维培养
3. **《密码学与网络安全》**：密码学原理
4. **《渗透测试实战》**：渗透测试方法论

### 官方文档
1. **OWASP**：[owasp.org](https://owasp.org/) - Web应用安全
2. **NIST网络安全**：[csrc.nist.gov](https://csrc.nist.gov/)
3. **CISA**：[cisa.gov](https://www.cisa.gov/) - 安全最佳实践

### 在线课程
1. **OWASP ZAP**：[zaproxy.org](https://www.zaproxy.org/) - 安全测试工具
2. **PortSwigger Web Security**：[portswigger.net](https://portswigger.net/web-security) - Web安全
3. **Offensive Security**：[offensive-security.com](https://www.offensive-security.com/) - 渗透测试

### 技术博客与案例
1. **Krebs on Security**：[krebsonsecurity.com](https://krebsonsecurity.com/) - 安全新闻
2. **Schneier on Security**：[schneier.com](https://www.schneier.com/) - 安全分析
3. **Google Security Blog**：[security.googleblog.com](https://security.googleblog.com/) - Google安全实践

### 开源项目参考
1. **OWASP ZAP**：[github.com/zaproxy/zaproxy](https://github.com/zaproxy/zaproxy) - 安全扫描
2. **Trivy**：[github.com/aquasecurity/trivy](https://github.com/aquasecurity/trivy) - 镜像扫描
3. **Falco**：[github.com/falcosecurity/falco](https://github.com/falcosecurity/falco) - 运行时安全
4. **Vault**：[github.com/hashicorp/vault](https://github.com/hashicorp/vault) - 密钥管理

### 权威标准
1. **OWASP Top 10**：[owasp.org/www-project-top-ten/](https://owasp.org/www-project-top-ten/)
2. **CWE**：[cwe.mitre.org](https://cwe.mitre.org/) - 通用缺陷枚举
3. **CVE**：[cve.mitre.org](https://cve.mitre.org/) - 漏洞数据库
4. **CIS Benchmarks**：[cisecurity.org](https://www.cisecurity.org/cis-benchmarks/)

## 学习产出要求

### 设计产出
1. ✅ 安全架构设计文档
2. ✅ 认证授权方案
3. ✅ 安全编码规范
4. ✅ 应急响应流程

### 代码产出
1. ✅ JWT认证服务
2. ✅ 输入验证模块
3. ✅ 安全头中间件
4. ✅ 日志与审计模块

### 技能验证
1. ✅ 理解常见安全威胁
2. ✅ 掌握密码学基础
3. ✅ 能够实现认证授权
4. ✅ 能够进行安全编码
5. ✅ 能够配置容器安全

### 文档产出
1. ✅ 安全检查清单
2. ✅ 漏洞修复指南
3. ✅ 安全运维手册

## 时间安排建议

### 第1天（安全基础与认证）
- **上午（4小时）**：安全基础概念
  - 安全三要素
  - 常见威胁
  - 防护原则

- **下午（4小时）**：密码学与认证
  - 加密算法
  - TLS/SSL
  - OAuth/JWT

- **晚上（2小时）**：应用安全
  - OWASP Top 10
  - 安全编码

### 第2天（网络安全与运维）
- **上午（4小时）**：网络安全
  - DDoS防护
  - WAF配置
  - TLS加固

- **下午（4小时）**：容器与云安全
  - 容器安全
  - K8s安全
  - 服务网格安全

- **晚上（2小时）**：监控与响应
  - 安全监控
  - 应急响应
  - 合规审计

## 学习方法建议

### 1. 理论与实践结合
- 学习安全原理
- 使用安全工具
- 进行漏洞修复
- 定期安全审计

### 2. 关注最新威胁
- 订阅安全公告
- 关注CVE漏洞
- 学习攻击手法
- 及时打补丁

### 3. 建立安全文化
- 安全代码规范
- 安全审查流程
- 安全意识培训
- 定期演练

## 常见问题与解决方案

### Q1：如何防止SQL注入？
**A**：防御方法：
- 使用参数化查询
- ORM框架自动防护
- 输入验证
- 最小权限原则

### Q2：如何安全存储密码？
**A**：存储方法：
- 使用bcrypt/Argon2
- 加盐处理（自动）
- 设置合理工作因子
- 禁止明文存储

### Q3：如何配置TLS？
**A**：配置要点：
- 使用TLS 1.3
- 强密码套件
- 证书自动化
- HSTS头启用

### Q4：如何进行安全监控？
**A**：监控策略：
- 收集认证日志
- 部署入侵检测
- 配置异常告警
- 定期日志审计

### Q5：容器安全怎么做？
**A**：安全措施：
- 最小基础镜像
- 镜像扫描
- 只读文件系统
- 资源限制
- 网络隔离

## 知识体系构建

### 核心知识领域

#### 1. 安全基础
```
安全基础
├── CIA三要素
│   ├── 机密性
│   ├── 完整性
│   └── 可用性
├── 常见威胁
│   ├── 注入攻击
│   ├── XSS/CSRF
│   ├── DDoS
│   └── 社会工程
└── 防护原则
    ├── 最小权限
    ├── 纵深防御
    └── 默认安全
```

#### 2. 密码学
```
密码学
├── 对称加密
│   └── AES
├── 非对称加密
│   ├── RSA
│   └── ECC
├── 哈希函数
│   ├── SHA-256
│   └── 密码哈希（bcrypt）
└── 数字证书
    └── X.509/TLS
```

#### 3. 应用安全
```
应用安全
├── OWASP Top 10
│   ├── 注入攻击
│   ├── 认证缺陷
│   └── 敏感数据泄露
├── 安全编码
│   ├── 输入验证
│   ├── 输出编码
│   └── 错误处理
└── 认证授权
    ├── OAuth 2.0
    ├── JWT
    └── RBAC
```

### 学习深度建议

#### 精通级别
- OWASP Top 10漏洞与防御
- TLS/SSL配置
- JWT认证实现
- 容器安全配置

#### 掌握级别
- 密码学原理
- OAuth 2.0流程
- K8s RBAC
- 安全监控与响应

#### 了解级别
- 渗透测试方法
- 取证分析
- 合规审计
- 硬件安全模块

## 下一步学习

### 立即进入
1. **容器安全实践**：
   - Trivy镜像扫描
   - Falco运行时防护
   - Vault密钥管理

2. **安全编码规范**：
   - ESLint安全插件
   - 代码审查清单

### 后续深入
1. **云原生安全**（路径06延伸）：服务网格安全
2. **渗透测试**：OWASP ZAP使用

### 持续跟进
- CVE漏洞公告
- OWASP更新
- 安全最佳实践

---

## 学习路径特点

### 针对人群
- 需要了解安全基础的工程师
- 面向JD中的"安全性"要求
- 适合需要构建安全系统的开发者

### 学习策略
- **中等强度**：2天集中学习，每天6-8小时
- **实践导向**：安全编码 + 工具使用
- **防御为主**：学习攻击手法，掌握防御方法

### 协同学习
- 与API开发：安全认证授权
- 与容器路径：容器安全
- 与运维路径：安全监控

### 质量保证
- 内容基于权威标准
- 代码符合安全规范
- 工具有效可验证

---

*学习路径设计：针对需要了解安全基础的工程师，掌握系统安全基础*
*时间窗口：春节第3周2天，中等强度学习系统安全*
*JD对标：满足JD中安全性、认证授权、容器安全等要求*
