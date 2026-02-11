# Python Type Hint 与 Mypy 深度解析

**日期**: 2026-02-10
**学习路径**: 02 - Python现代化开发
**对话主题**: Python 3.11/3.12 类型系统新特性、Type Hint 原理与 Mypy 工作机制

## 一、Python 3.11 新增类型特性

### 1. `Self` 类型 - 更优雅的链式调用

**解决的问题：**

```python
# 不用 Self 的问题：需要字符串前向引用
class OldBuilder:
    def set_name(self, name: str) -> "OldBuilder":
        self.name = name
        return self

# 子类继承时返回类型仍然是父类
old_builder = OldBuilder().set_name("test")
# old_builder 的类型是 OldBuilder，不是子类

# 如果子类有额外方法，无法链式调用
class IntBuilder(OldBuilder):
    def set_value(self, value: int) -> "IntBuilder":
        self.value = value
        return self

# ❌ 这样才行
int_builder = IntBuilder().set_name("test").set_value(42)
# 但 set_name 返回 OldBuilder，不是 IntBuilder
```

**使用 Self 后的改进：**

```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self

class IntBuilder(Builder):
    def set_value(self, value: int) -> Self:
        self.value = value
        return self

# ✅ 子类链式调用类型正确
int_builder = IntBuilder().set_name("test").set_value(42)
# int_builder 的类型是 IntBuilder
```

**`Self` 的实现原理：**

```python
# Self 类型在运行时的表示（简化理解）
class _Self:
    """Self 无法直接实例化，只用于类型注解"""
    def __init__(self):
        raise TypeError("Self cannot be instantiated")

# 类型检查器处理流程：
"""
1. 解析 Self 返回类型
2. 确定实际调用时的类（this 或子类）
3. 将 Self 替换为实际类型
4. 进行类型检查
"""

# 具体示例
class Parent:
    def chain(self) -> Self:
        return self

class Child(Parent):
    def extra(self) -> int:
        return 42

child = Child().chain()
# child.chain() 返回 Self → Child
# child 的类型是 Child
# child.extra()  # ✅ OK
```

### 2. `Never` 类型 - 永不返回的精确表示

**`Never` vs `NoReturn`：**

```python
from typing import NoReturn, Never

# NoReturn - 用于标记永不返回的函数
def fail_noReturn() -> NoReturn:
    raise RuntimeError("error")

def infinite_noReturn() -> NoReturn:
    while True:
        pass

# Never - Python 3.11+ 等价于 NoReturn，语义更清晰
def fail_never() -> Never:
    raise RuntimeError("error")

# 类型检查器对 Never 的处理
def test_never():
    x: int = fail_never()  # ❌ Type error: Never 不能赋值给任何类型
    print("unreachable")   # ⚠️ 类型检查器知道这行永远不会执行

# Never 在联合类型收缩中的作用
def narrow(x: str | int) -> None:
    if isinstance(x, str):
        print(x.upper())
    else:
        # 在这个分支，x 的类型是 Never
        # 因为 str | int 在 isinstance 后，
        # else 分支不可能到达
        reveal_type(x)  # Revealed type is: Never
```

**为什么需要 Never？**

```python
# 1. 类型理论上的"底类型"（bottom type）
# 2. 更精确的类型收缩
# 3. 与 NoReturn 语义区分：
#    - NoReturn: 函数永不返回
#    - Never: 类型是空的，不可能存在

from typing import Union, Literal

# 使用 Never 进行穷尽检查
def exhaustive(x: Literal["a", "b"]) -> str:
    if x == "a":
        return "is a"
    elif x == "b":
        return "is b"
    else:
        # x 在这里是 Never（类型检查器知道所有情况已覆盖）
        reveal_type(x)  # Never
        raise AssertionError("Unreachable")
```

### 3. `TypeGuard` 增强 - 用户自定义类型谓词

**基础用法：**

```python
from typing import TypeGuard

def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    """检查是否为字符串列表"""
    return all(isinstance(x, str) for x in val)

def process(items: list[object]) -> None:
    if is_string_list(items):
        # ✅ 类型检查器知道 items 是 list[str]
        for item in items:
            print(item.upper())  # str 的方法可用
            print(item + 1)      # ❌ Type error: 不支持 str + int
    else:
        # items 仍然是 list[object]
        pass
```

**Python 3.11 的改进：**

```python
# 3.11 前：TypeGuard 必须精确匹配
def is_str_seq(seq: list[object]) -> TypeGuard[list[str]]:
    ...

# 3.11+：TypeGuard 可以返回协变类型
from typing import Sequence

def is_str_sequence(seq: Sequence[object]) -> TypeGuard[Sequence[str]]:
    """
    Sequence[str] 是 Sequence[object] 的子类型
    3.11 前不允许，现在可以
    """
    return all(isinstance(x, str) for x in seq)
```

**TypeGuard 的实现原理：**

```python
# TypeGuard 本质是标记类型检查函数的返回类型
# 告诉类型检查器：返回 True 时，缩小参数类型

def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

# 类型检查器的处理：
"""
1. 分析函数体（isinstance 检查）
2. 如果返回 True，参数类型被收窄为 list[str]
3. 如果返回 False，收窄不生效
"""

# 注意事项：TypeGuard 不会影响 else 分支
def process(items: list[object]) -> None:
    if is_string_list(items):
        # items: list[str]
        pass
    else:
        # items: list[object]（不是 list[str] 的补集）
        # 这是 TypeGuard 和 isinstance 的区别
        pass
```

## 二、Python 3.12 新增类型特性 (PEP 695)

### 1. `type` 语句 - 原生类型别名

**旧写法 vs 新写法：**

```python
# Python 3.12 前：使用 TypeAlias
from typing import TypeAlias, Literal

UserId: TypeAlias = int
Status: TypeAlias = Literal["pending", "done", "cancelled"]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

# Python 3.12+：使用 type 语句
type UserId = int
type Status = Literal["pending", "done", "cancelled"]
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
```

**type 语句的优势：**

```python
# 1. 泛型参数在类型定义左侧，更清晰
# 旧写法
from typing import TypeAlias, Callable, TypeVar

T = TypeVar("T")
Result: TypeAlias = Callable[[T], T]

# 新写法
type Result[T] = Callable[[T], T]

# 2. 支持默认值
type Callback[T = str] = Callable[[T], None]

def on_click(callback: Callback) -> None:
    callback("hello")  # 默认 str

def on_click_int(callback: Callback[int]) -> None:
    callback(42)

# 3. 使用 | 操作符更自然
type Maybe[T] = T | None  # ✅
# vs
Maybe: TypeAlias = Optional[T]  # 需要导入 Optional
```

**泛型别名的复杂用法：**

```python
# 带约束的泛型
T = TypeVar("T", int, float)

type Numeric = T  # T 必须是 int 或 float

# 多重界定
U = TypeVar("U", bound=int)
type ConstrainedInt = U

# 使用泛型别名
type DictOf[K, V] = dict[K, V]

def process(d: DictOf[str, int]) -> None:
    pass

process({"a": 1, "b": 2})  # ✅
```

### 2. 泛型类的简化语法

**不再需要继承 Generic：**

```python
# Python 3.12 前
from typing import TypeVar, Generic

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value

    def set(self, value: T) -> None:
        self.value = value

# Python 3.12+
class Box[T]:  # 直接使用类型参数
    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value

    def set(self, value: T) -> None:
        self.value = value

# 多类型参数
class KeyValue[K, V]:
    def __init__(self, key: K, value: V):
        self.key = key
        self.value = V

# 继承和实现
class Container[T]:
    def get(self) -> T: ...

class StringContainer(Container[str]):
    def get(self) -> str:
        return "hello"
```

### 3. 泛型默认值

```python
# 带默认值的泛型
type Result[T, E = Exception] = T | E

# 使用默认类型
def ok(value: int) -> Result[int]:
    return value

# 覆盖默认类型
def fail(error: str) -> Result[str, ValueError]:
    raise ValueError(error)

# 多参数，部分默认值
type Pair[T, U = tuple[T, T]] = U

# 使用
p1: Pair[int] = (1, 2)              # U 默认是 tuple[int, int]
p2: Pair[int, list[int]] = [1, 2]   # 覆盖 U
```

**与类结合：**

```python
class Repository[T = int]:  # 默认 T 为 int
    def get(self, id: T) -> dict | None:
        ...

# 使用默认
repo: Repository = Repository()
repo.get(1)  # id 是 int

# 指定类型
str_repo: Repository[str] = Repository()
str_repo.get("abc")  # id 是 str
```

## 三、Type Hint 的工作原理

### 1. 类型注解的存储与访问

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

# 注解存储在 __annotations__ 中
print(greet.__annotations__)
# {'name': <class 'str'>, 'return': <class 'str'>}

# 访问具体注解
print(greet.__annotations__["name"])  # <class 'str'>

# 类属性注解
class User:
    name: str
    age: int

print(User.__annotations__)
# {'name': <class 'str'>, 'age': <class 'int'>}
```

### 2. `from __future__ import annotations` 的影响

```python
from __future__ import annotations

# 注解变成字符串（延迟求值），不是实际类型对象
def foo(x: MyClass) -> MyClass:
    pass

print(foo.__annotations__)
# {'x': 'MyClass', 'return': 'MyClass'}
# 注意：是字符串 'MyClass'，不是类对象

# 需要字符串前向引用的场景
class Node:
    def __init__(self, value: int, next: "Node | None" = None):
        self.value = value
        self.next = next

# 使用 future import 后不需要引号
class Node:
    def __init__(self, value: int, next: Node | None = None):
        self.value = value
        self.next = next  # ✅ Node 可以直接使用
```

**运行时获取注解类型的正确方式：**

```python
from __future__ import annotations

def get_type_hints(func):
    """安全获取类型提示（处理字符串化注解）"""
    import typing
    hints = func.__annotations__.copy()
    for name, hint in hints.items():
        if isinstance(hint, str):
            # 动态求值字符串注解
            hints[name] = eval(hint, func.__globals__)
    return hints

# 示例
def process(x: int, y: str) -> bool:
    pass

hints = get_type_hints(process)
print(hints)
# {'x': <class 'int'>, 'y': <class 'str'>, 'return': <class 'bool'>}
```

### 3. 类型检查器的注解处理流程

```python
# 类型检查器（如 mypy）处理流程：
"""
1. 解析：读取源码，识别类型注解
2. 语义分析：解析类型表达式，处理 imports
3. 类型推断：根据上下文推断变量类型
4. 类型检查：验证类型一致性
5. 错误报告：输出类型错误位置和原因
"""

# 示例：类型检查器如何处理这个函数
def add(a: int, b: str) -> int:
    return a + b  # ❌ Type error

"""
mypy 处理过程：
1. 解析：a: int, b: str, return: int
2. 分析 a + b 操作
3. 查找 int.__add__(str) 的支持
4. 发现不兼容：unsupported operand types for +
5. 报告错误：
   error: Unsupported operand types for + ("int" and "str")
   note: Following member(s) of left operand have conflicts:
         __add__: "int" has no attribute "__add__" defined in "int"
"""
```

## 四、Mypy 的工作原理

### 1. Mypy 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Mypy Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   源码 ──► 解析器 ──► AST ──► 语义分析 ──► 类型检查 ──► 输出  │
│                │                    │                         │
│                ▼                    ▼                         │
│           FastParser           SemanticAnalyzer              │
│                              (处理 imports/annotations)      │
│                                                              │
│   Checker ◄───────────────────────────────────────────── TypeAnalyzer │
│   (具体类型规则)              (类型推断和收窄)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

# 主要组件：
# 1. FastParser - 解析 Python 代码生成 AST
# 2. SemanticAnalyzer - 处理 imports、注解求值、前向引用
# 3. TypeAnalyzer - 类型推断和收窄
# 4. Checker - 具体类型规则检查
```

### 2. 类型推断机制

```python
# mypy 使用多种推断策略：

# 1. 字面量推断
x = 10        # x: int
x = "hello"   # x: str（类型变更）

# 2. 函数参数推断
def foo(x, y):  # x: Any, y: Any（无注解）
    return x + y  # Any + Any 合法

# 3. 上下文推断
x: int = 10    # 显式注解优先
x = 10         # 根据右侧推断: int

# 4. 控制流分析（类型收窄）
from typing import Literal

def process(x: int | str | None) -> None:
    if x is None:
        reveal_type(x)  # None
    elif isinstance(x, str):
        reveal_type(x)  # str
    elif isinstance(x, int):
        reveal_type(x)  # int
    else:
        reveal_type(x)  # Never（所有情况已覆盖）

# 5. 字面量推断
status: Literal["ok", "error"] = "ok"
# status = "unknown"  # ❌ Type error
```

### 3. 协变与逆变

```python
class Animal:
    pass

class Dog(Animal):
    def bark(self) -> str:
        return "Woof!"

# 返回类型协变（Producer 模式）
def get_dog() -> Dog:
    return Dog()

def get_animal() -> Animal:
    return Animal()

# Callable 的协变/逆变规则：
# - 返回类型协变：Callable[[Dog], Animal] = Callable[[Dog], Dog] ✅
# - 参数类型逆变：Callable[[Animal], None] = Callable[[Dog], None] ✅

# 协变示例
T_co = TypeVar("T_co", covariant=True)

class Producer[T_co]:
    def produce(self) -> T_co:
        ...

producer_dog: Producer[Dog] = Producer()  # 创建实例
producer_animal: Producer[Animal] = producer_dog  # ✅ OK
# 因为 Producer[Dog] 是 Producer[Animal] 的子类型

# 逆变示例
T_contra = TypeVar("T_contra", contravariant=True)

class Consumer[T_contra]:
    def consume(self, value: T_contra) -> None:
        ...

consumer_animal: Consumer[Animal] = Consumer()
consumer_dog: Consumer[Dog] = consumer_animal  # ✅ OK
# 因为 Consumer[Animal] 是 Consumer[Dog] 的父类型
# 接受 Animal 的可以接受 Dog
```

### 4. 配置文件与严格模式

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true                    # 开启所有严格检查
warn_return_any = true           # 返回 Any 时警告
warn_unused_ignores = true       # 未使用的 type: ignore 警告
disallow_untyped_defs = true     # 不允许无类型注解的函数
disallow_any_generics = true     # 不允许泛型使用 Any
check_untyped_defs = true        # 检查无类型注解的函数体
warn_redundant_casts = true      # 冗余 cast 警告
disallow_untyped_calls = true    # 不允许调用无类型注解的函数

# 第三方库配置
[[tool.mypy.overrides]]
module = "numpy.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "pandas.*"
ignore_missing_imports = true

# 本地包配置
[[tool.mypy.overrides]]
module = "mypackage.*"
strict = true
```

### 5. 常见 Mypy 错误与修复

```python
from typing import Optional, Any

# 错误 1: Implicit optional
def foo(x: str = None) -> str:  # ❌ None 不能赋值给 str
    return x or ""

# 修复
def foo(x: Optional[str] = None) -> str:  # ✅
    return x or ""

# 或使用 | 语法（Python 3.10+）
def foo(x: str | None = None) -> str:  # ✅
    return x or ""

# 错误 2: Untyped def
@decorator
def bar(x):  # ❌ 没有类型注解
    return x

# 修复 1：添加注解
@decorator
def bar(x: int) -> int:  # ✅
    return x

# 修复 2：stub 文件
# mymodule.pyi
from typing import Callable, TypeVar
T = TypeVar("T")
def decorator(func: Callable[..., T]) -> Callable[..., T]: ...

# 错误 3: Incompatible return type
def get_value() -> int:
    return "hello"  # ❌ str 不能赋值给 int

# 修复
def get_value() -> str:
    return "hello"  # ✅

# 错误 4: No overload variant matches
from typing import overload

@overload
def process(x: int) -> int: ...
@overload
def process(x: str) -> str: ...
def process(x):  # ❌ 实现函数被检查时报错
    return x

# 修复：添加类型注解到实现
@overload
def process(x: int) -> int: ...
@overload
def process(x: str) -> str: ...
def process(x: int | str) -> int | str:  # ✅
    return x

# 错误 5: Unused "type: ignore"
x = 10  # type: ignore  # ❌ 这个 ignore 没有必要

# 修复：移除
x = 10  # ✅
```

### 6. Mypy 插件系统

```python
# mypy 插件示例：为自定义 dataclass 生成 __init__ 类型

# mypkg/plugin.py
from mypy.plugin import Plugin
from mypy.plugins.common import add_method_to_class
from mypy.types import TypeOfAny, AnyType

class MyPlugin(Plugin):
    def get_class_decorator_hook(self, fullname):
        if fullname == 'mypackage.my_dataclass':
            return self._infer_dataclass_init
        return None

    def _infer_dataclass_init(self, ctx):
        # 1. 获取 dataclass 字段
        fields = self.get_dataclass_fields(ctx.cls)

        # 2. 生成 __init__ 方法签名
        args = []
        for field in fields:
            arg_type = self.build_type_from_annotation(field.type)
            args.append((field.name, arg_type, field.default))

        # 3. 添加 __init__ 方法
        add_method_to_class(ctx.api, ctx.cls, '__init__', args)
        return None

def plugin(version):
    return MyPlugin

# 使用插件
# pyproject.toml
[tool.mypy]
plugins = ["mypkg.plugin"]
```

## 五、实战技巧

### 1. 渐进式类型添加策略

```python
# 阶段 1：从核心模块开始
# - 业务逻辑层
# - 数据模型
# - API 接口

# 阶段 2：中间层
# - 服务层
# - 仓储层

# 阶段 3：工具函数
# - 辅助函数
# - 通用模块

# 使用 reveal_type() 调试
def some_function():
    x = complicated_expression()
    reveal_type(x)  # mypy 会显示实际推断类型

    if isinstance(x, str):
        reveal_type(x)  # 收窄后的类型
```

### 2. 与 FastAPI 深度集成

```python
from fastapi import FastAPI, Depends, Query, Path, Body
from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator

app = FastAPI()

# FastAPI 利用类型注解的方式：
# 1. 路由参数自动解析和验证
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,                    # 路径参数：自动转换 int
    include_extra: bool = False,     # 查询参数：自动转换 bool
    page: int = Query(default=1, ge=1),  # 查询参数 + 验证
) -> dict[str, str]:
    return {"user_id": str(user_id)}

# 2. 请求体验证
class UserCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    email: Annotated[str, Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")]
    age: int | None = None

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, v: str) -> str:
        return v.lower()

@app.post("/users", status_code=201)
async def create_user(user: UserCreate) -> UserCreate:
    # user 已经过验证
    return user

# 3. 依赖注入
async def get_db():
    ...

@app.get("/items")
async def get_items(
    db: Annotated[Session, Depends(get_db)]
):
    ...

# 4. 联合类型 + 字面量
@app.get("/orders/{status}")
async def get_orders(
    status: Literal["pending", "processing", "completed"]
) -> list[dict]:
    ...
```

### 3. 高级模式

**Result 类型（避免异常）：**

```python
from typing import Generic, TypeVar, Literal

T = TypeVar("T")
E = TypeVar("E", bound=Exception)

class Ok(Generic[T]):
    def __init__(self, value: T):
        self.value = value

class Err(Generic[E]):
    def __init__(self, error: E):
        self.error = error

Result = Ok[T] | Err[E]

def safe_divide(a: int, b: int) -> Result[int, ZeroDivisionError]:
    if b == 0:
        return Err(ZeroDivisionError("Division by zero"))
    return Ok(a / b)

result = safe_divide(10, 2)
if isinstance(result, Ok):
    print(result.value)  # 5.0
else:
    print(result.error)  # 处理错误
```

**Builder 模式（链式调用）：**

```python
from typing import Self

class RequestBuilder:
    _url: str
    _method: str
    _headers: dict[str, str]

    def url(self, url: str) -> Self:
        self._url = url
        return self

    def method(self, method: str) -> Self:
        self._method = method
        return self

    def header(self, key: str, value: str) -> Self:
        self._headers[key] = value
        return self

    def build(self) -> dict:
        return {
            "url": self._url,
            "method": self._method,
            "headers": self._headers
        }

# 使用
request = (RequestBuilder()
    .url("https://api.example.com")
    .method("POST")
    .header("Content-Type", "application/json")
    .build())

# 类型检查器知道 request 的类型
reveal_type(request)  # dict[str, Any]
```

**Repository 模式（协议定义）：**

```python
from typing import Protocol, TypeVar

T = TypeVar("T")

class Repository(Protocol[T]):
    """Repository 协议"""
    def get(self, id: int) -> T | None: ...
    def save(self, entity: T) -> T: ...
    def delete(self, id: int) -> None: ...

class User:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

class UserRepository:
    def get(self, id: int) -> User | None:
        # 实现
        ...

    def save(self, entity: User) -> User:
        # 实现
        ...

    def delete(self, id: int) -> None:
        # 实现
        ...

def process_repo(repo: Repository[User]) -> None:
    user = repo.get(1)
    if user:
        repo.save(user)

process_repo(UserRepository())  # ✅ UserRepository 实现了 Repository[User]
```

## 六、Python 3.11/3.12 类型特性对比

| 特性 | Python 3.11 | Python 3.12 | 说明 |
|------|-------------|-------------|------|
| `Self` | ✅ 原生 | ✅ | 链式调用返回自身类型 |
| `Never` | ✅ 原生 | ✅ | 永不返回/底类型 |
| `TypeGuard` 协变 | ✅ 改进 | ✅ | 支持协变返回类型 |
| `type` 语句 | ❌ | ✅ | 原生类型别名 |
| 泛型类简写 | ❌ | ✅ | 不需要继承 Generic |
| 泛型默认值 | ❌ | ✅ | 支持默认类型参数 |

## 学习笔记

1. **Python 类型系统设计哲学**：渐进式、可选、工具驱动
2. **3.12 的 type 语句**：语言层面的原生支持，语法更简洁
3. **Self 类型**：完美解决链式调用和子类继承的返回类型问题
4. **Mypy 工作流程**：解析 → 语义分析 → 类型推断 → 检查 → 报告
5. **协变/逆变**：理解子类型关系的关键，Function 类型参数逆变、返回类型协变
6. **FastAPI 深度集成**：类型注解不仅用于检查，还用于验证、文档生成、依赖注入

## 后续行动计划

1. 在项目中实践 `type` 语句和泛型默认值
2. 深入理解 Mypy 插件机制
3. 学习 pyright（微软出品，更严格的类型检查）
4. 探索类型系统在运行时反射中的应用

## 参考资料

- [PEP 673 - Self Type](https://peps.python.org/pep-0673/)
- [PEP 654 - TypeGuard](https://peps.python.org/pep-0654/)
- [PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)
- [Mypy 文档](https://mypy.readthedocs.io/)
- [Pyright 文档](https://microsoft.github.io/pyright/)
