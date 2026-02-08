# 实践项目：全栈TypeScript应用

## 项目概述
- **目标**：构建一个基于React和Node.js的现代化全栈应用
- **技术栈**：TypeScript + React + Node.js (Express/NestJS) + PostgreSQL + Prisma
- **时间**：春节第2周前4天（与TypeScript学习并行）
- **关联学习**：TypeScript全栈学习

## 项目背景
作为全栈工程师，能够独立完成从前端UI到后端API再到数据库的完整开发流程是基本要求。本项目旨在通过构建一个"任务与知识管理系统"，通过实际编码掌握TypeScript在全栈开发中的应用，以及现代前端工程化和后端架构设计。

## 功能需求

### 核心功能
1. **用户认证**：注册、登录、JWT认证、角色管理
2. **任务管理**：看板视图（Kanban）、任务CRUD、拖拽排序
3. **知识库**：Markdown编辑器、文档树状结构、全文搜索
4. **个人中心**：个人资料修改、设置、操作日志

### 高级功能（可选）
1. **实时协作**：WebSocket实现的多人实时编辑
2. **AI辅助**：集成简单AI接口进行文本润色
3. **暗色模式**：系统级主题切换
4. **数据可视化**：任务统计图表

## 技术架构

### 整体架构
```
浏览器 (React SPA) 
      ↕ (REST/GraphQL)
API网关 (Nginx/Node)
      ↕
后端服务 (Node.js/Express) ↔ 数据库 (PostgreSQL)
      ↕
   缓存 (Redis)
```

### 技术选型
- **前端**：
  - 框架：React 18 + Vite
  - 语言：TypeScript
  - 路由：React Router v6
  - 状态管理：Zustand 或 Redux Toolkit
  - UI组件库：Ant Design 或 Tailwind CSS + Headless UI
  - 数据请求：Axios 或 React Query

- **后端**：
  - 运行时：Node.js
  - 框架：Express 或 NestJS (推荐NestJS以匹配强类型风格)
  - ORM：Prisma
  - 验证：Zod 或 class-validator

## 实现方案

### 1. 数据库设计 (Prisma Schema)
```prisma
// schema.prisma

model User {
  id        String   @id @default(uuid())
  email     String   @unique
  password  String
  name      String?
  role      Role     @default(USER)
  tasks     Task[]
  docs      Document[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}

model Task {
  id          String   @id @default(uuid())
  title       String
  description String?
  status      TaskStatus @default(TODO)
  priority    Priority   @default(MEDIUM)
  userId      String
  user        User     @relation(fields: [userId], references: [id])
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
}

enum Role {
  USER
  ADMIN
}

enum TaskStatus {
  TODO
  IN_PROGRESS
  DONE
}
```

### 2. 后端API实现 (Express + TypeScript)
```typescript
// src/controllers/task.controller.ts

import { Request, Response } from 'express';
import { prisma } from '../lib/prisma';
import { CreateTaskSchema } from '../schemas/task.schema';

export const createTask = async (req: Request, res: Response) => {
  try {
    // 验证请求数据
    const data = CreateTaskSchema.parse(req.body);
    
    // 获取当前用户ID (从中间件注入)
    const userId = req.user?.id;

    const task = await prisma.task.create({
      data: {
        ...data,
        userId
      }
    });

    res.json(task);
  } catch (error) {
    res.status(400).json({ error: 'Invalid data' });
  }
};
```

### 3. 前端组件实现 (React + TypeScript)
```tsx
// src/components/TaskList.tsx

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { getTasks } from '../api/tasks';
import { Task } from '../types';

export const TaskList: React.FC = () => {
  const { data: tasks, isLoading, error } = useQuery<Task[]>({
    queryKey: ['tasks'],
    queryFn: getTasks
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error loading tasks</div>;

  return (
    <div className="grid gap-4">
      {tasks?.map(task => (
        <div key={task.id} className="p-4 border rounded shadow">
          <h3 className="font-bold">{task.title}</h3>
          <p className="text-gray-600">{task.description}</p>
          <span className={`badge ${task.status}`}>{task.status}</span>
        </div>
      ))}
    </div>
  );
};
```

## 项目结构

```
fullstack-app/
├── packages/
│   ├── client/              # 前端应用
│   │   ├── src/
│   │   │   ├── api/
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── pages/
│   │   │   └── stores/
│   │   ├── package.json
│   │   └── vite.config.ts
│   ├── server/              # 后端应用
│   │   ├── src/
│   │   │   ├── config/
│   │   │   ├── controllers/
│   │   │   ├── middlewares/
│   │   │   ├── routes/
│   │   │   └── services/
│   │   ├── prisma/
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── shared/              # 前后端共享类型
│       └── src/
│           └── types.ts
├── package.json             # Monorepo配置
└── docker-compose.yml       # 数据库和其他服务
```

## 开发计划

### 第1天：环境搭建与基础架构
1. **上午**：Monorepo环境搭建
   - 配置pnpm workspace
   - 初始化Client和Server项目
   - 配置TypeScript共享配置

2. **下午**：后端基础开发
   - 设计数据库Schema
   - 配置Prisma和PostgreSQL
   - 实现用户认证API (JWT)

3. **晚上**：前端基础架构
   - 配置Vite + React
   - 集成Tailwind CSS
   - 封装Axios请求拦截器

### 第2天：核心业务功能
1. **上午**：后端CRUD开发
   - 实现任务管理API
   - 编写API单元测试
   - 实现数据验证逻辑

2. **下午**：前端业务组件
   - 开发任务列表组件
   - 实现任务创建/编辑表单
   - 集成React Query管理状态

3. **晚上**：前后端联调
   - 调试接口交互
   - 优化错误处理
   - 完善Loading状态

### 第3天：进阶功能与优化
1. **上午**：知识库功能
   - 集成Markdown编辑器
   - 实现文档存储接口
   - 优化编辑器体验

2. **下午**：性能与体验
   - 实现后端分页和过滤
   - 前端路由懒加载
   - 增加简单的Dashboard图表

3. **晚上**：部署准备
   - 编写Dockerfile
   - 配置Docker Compose一键启动
   - 编写README文档

## 学习收获

### 技术技能
1. ✅ 精通TypeScript在全栈中的应用（类型共享、泛型等）
2. ✅ 掌握React 18新特性和Hooks最佳实践
3. ✅ 熟悉Node.js后端开发模式和ORM使用
4. ✅ 了解现代前端工程化配置

### 架构能力
1. ✅ 全栈Monorepo架构设计经验
2. ✅ 前后端分离的API设计规范
3. ✅ 数据库模型设计能力
4. ✅ 统一的错误处理和状态管理方案

## 扩展方向

1. **迁移到Next.js**：将前端迁移到SSR框架，提升SEO和首屏性能
2. **GraphQL改造**：使用Apollo Server + Client 替换REST API
3. **微服务拆分**：将任务服务和文档服务拆分为独立微服务
4. **CI/CD流水线**：使用GitHub Actions实现自动构建和测试

---

*项目特点：贴近实际工作场景的全栈应用开发，强调TypeScript类型安全和工程化*
*学习价值：打通前后端开发链路，提升独立交付能力*
