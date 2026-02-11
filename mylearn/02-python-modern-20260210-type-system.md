# Python 3 现代类型系统深度解析

**日期**: 2026-02-10
**学习路径**: 02 - Python现代化开发
**对话主题**: Python 3 现代类型系统

## 问题背景

深入了解 Python 3 的现代类型系统，包括：
- 常见好用的类型注解
- 类型系统中的坑和注意事项
- 类型检查 linter 工具
- 性能、特性、运行时行为
- 特殊用法和高级特性
- FastAPI/Uvicorn 等框架的类型支持

## 一、类型系统演进与核心概念

### 版本演进时间线

```
Python 3.5 (2015)  → typing 模块首次引入 (PEP 484)
Python 3.6 (2016)  → 变量注解语法 (PEP 526)
Python 3.7 (2018)  → dataclass + from __future__ import annotations
Python 3.8 (2019)  → Final/Literal + typing_extensions
Python 3.9 (2020)  → 内置泛型 (list[str] 替代 List[str])
Python 3.10 (2021) → X | Y 联合语法 (PEP 604) + ParamSpec
Python 3.11 (2022) → Self + Never + TypeGuard (部分在 typing_extensions)
Python 3.12 (2023) → 类型别名语句 + 泛型类 (PEP 695)
```

### 核心哲学

Python 类型系统是**渐进式**的：
- **可选的**：不强制要求类型注解
- **运行时忽略**：默认不进行运行时类型检查（除非使用 `@runtime_checkable`）
- **工具驱动**：通过 mypy/pyright/pyre 等静态检查器发挥作用

## 二、现代类型注解（Python 3.9+）

### 内置泛型（无需导入 typing）

```python
# ✅ Python 3.9+ 推荐：直接使用内置泛型
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

def get_config() -> dict[str, str | int | bool]:
    return {"timeout": 30, "debug": True}

# ❌ 旧写法（3.9前需要，现在不推荐）
from typing import List, Dict, Union
def process_items_old(items: List[str]) -> Dict[str, int]:
    ...
```

### 联合类型的新语法（Python 3.10+）

```python
# ✅ Python 3.10+：使用 | 操作符
def handle(data: str | int | None) -> str:
    if data is None:
        return "empty"
    return str(data)

# ❌ 旧写法：Union[X, Y]
from typing import Union
def handle_old(data: Union[str, int, None]) -> str:
    ...

# 可选类型的简化
def foo(x: str | None = None):  # 等同于 Optional[str]
    pass
```

## 三、常见好用类型详解

### 1. NewType：运行时无开销的语义标记

```python
from typing import NewType

UserId = NewType('UserId', int)
ProductId = NewType('ProductId', int)

def get_user(user_id: UserId) -> str:
    return f"User {user_id}"

# 类型检查器会捕获错误
user_id: UserId = UserId(123)
product_id: ProductId = ProductId(456)

get_user(user_id)      # ✅ OK
get_user(product_id)   # ❌ Type error: expected UserId, got ProductId
get_user(123)          # ❌ Type error: expected UserId, got int

# 运行时：UserId(123) 返回 123，无任何开销
```

#### NewType 无开销原理深度解析

**NewType 的本质是什么？**

```python
# 查看 NewType 的源码实现（简化版）
def NewType(name, tp):
    """这是一个工厂函数，返回一个 callable"""
    def new_type(x):
        return x  # ⚠️ 直接返回原值，没有任何包装！

    new_type.__name__ = name
    new_type.__supertype__ = tp
    return new_type

# 当你调用 UserId(123) 时：
# 1. 调用 new_type(123)
# 2. 直接返回 123（不是 UserId 实例，就是 int 123）
# 3. 运行时类型仍然是 int
```

**运行时验证：**

```python
from typing import NewType

UserId = NewType('UserId', int)
UserId2 = NewType('UserId2', int)

# 创建实例
user_id = UserId(123)

# 运行时类型检查
print(type(user_id))           # <class 'int'> - 不是 UserId！
print(isinstance(user_id, int))  # True
print(user_id == 123)          # True - 直接比较
print(user_id + 1)             # 124 - 直接运算

# UserId 和 UserId2 创建的值可以互相赋值（运行时）
user_id2 = UserId2(user_id)  # ✅ 运行时 OK
# 但类型检查器会报错：❌ Type error

# 内存地址完全相同
original = 123
wrapped = UserId(123)
print(id(original) == id(wrapped))  # True - 同一个对象！
```

**类型检查器如何看待 NewType？**

```python
from typing import NewType

UserId = NewType('UserId', int)
ProductId = NewType('ProductId', int)

# 类型检查器的处理：
# 1. UserId 被视为 int 的"子类型"
# 2. 但不是普通子类型，是"名义子类型"（nominal subtype）
# 3. 即使底层类型相同，UserId != ProductId

def process_user(id: UserId) -> None: ...

# 类型检查器会分析：
# process_user(UserId(123))    ✅ UserId → UserId (完全匹配)
# process_user(ProductId(456)) ❌ ProductId ≠ UserId (名义不同)
# process_user(123)            ❌ int ≠ UserId (需要显式转换)
```

**为什么这样设计？权衡与哲学**

```python
# 设计目标 1：零运行时开销 ✅
# NewType 不创建任何新对象，只是"类型别名"的增强版

# 设计目标 2：静态类型区分 ✅
# 防止混淆语义不同的值
Dollars = NewType('Dollars', float)
Euros = NewType('Euros', float)

def add_dollars(a: Dollars, b: Dollars) -> Dollars:
    return Dollars(a + b)

add_dollars(Dollars(10.0), Dollars(5.0))    # ✅ OK
# add_dollars(Euros(10.0), Dollars(5.0))   # ❌ Type error

# 设计目标 3：可选的渐进式类型 ✅
# 不使用类型检查时，代码完全正常工作
value = UserId(123)
print(value * 2)  # 246 - 运行时完全正常
```

**与类型别名的区别**

```python
# 类型别名
UserIdAlias = int  # 只是另一个名字

# NewType
UserId = NewType('UserId', int)  # 创建新类型

# 类型检查器的区别：
def f1(x: UserIdAlias) -> None: ...  # UserIdAlias 就是 int
def f2(x: UserId) -> None: ...       # UserId 是独特类型

f1(123)        # ✅ OK (UserIdAlias = int)
# f2(123)      # ❌ Type error (需要 UserId)
f2(UserId(123)) # ✅ OK
```

**实际应用模式**

```python
from typing import NewType

# 1. ID 类型（防止混淆）
UserId = NewType('UserId', int)
OrderId = NewType('OrderId', int)
ProductId = NewType('ProductId', int)

def get_user(user_id: UserId) -> str: ...
def get_order(order_id: OrderId) -> str: ...

# 2. 防止单位错误
Meters = NewType('Meters', float)
Seconds = NewType('Seconds', float)

def calculate_speed(distance: Meters, time: Seconds) -> float:
    return distance / time

# calculate_speed(Meters(100), Meters(10))  # ❌ Type error
calculate_speed(Meters(100), Seconds(10))   # ✅ OK

# 3. 验证标记
ValidatedEmail = NewType('ValidatedEmail', str)

def send_email(email: ValidatedEmail, message: str) -> None:
    # 调用者保证 email 已经验证过
    pass

def validate_email(email: str) -> ValidatedEmail:
    if '@' not in email:
        raise ValueError("Invalid email")
    return ValidatedEmail(email)

# 使用
raw_email = "user@example.com"
validated = validate_email(raw_email)
send_email(validated, "Hello")  # ✅ OK
# send_email(raw_email, "Hello")  # ❌ Type error - 未验证
```

**性能对比测试**

```python
import timeit
from typing import NewType

UserId = NewType('UserId', int)

N = 10_000_000

# 直接使用 int
t1 = timeit.timeit(lambda: x + 1 for x in range(N), number=N)

# 使用 NewType
t2 = timeit.timeit(lambda: UserId(x) + 1 for x in range(N), number=N)

# 结果：t1 ≈ t2，性能差异可忽略不计
```

**总结：NewType 的实现机制**

| 方面 | 实现方式 | 说明 |
|------|----------|------|
| **运行时** | `def f(x): return x` | 恒等函数，直接返回 |
| **内存** | 零开销 | 不创建新对象 |
| **类型检查器** | 名义类型区分 | 视为独特类型 |
| **继承关系** | 视为子类型 | UserId <: int |
| **可逆性** | 不可逆 | int ≠ UserId |

### 2. Literal：精确的枚举替代品

```python
from typing import Literal

# 限定只有这几个字面值
Status = Literal["pending", "processing", "completed", "failed"]

def update_status(status: Status) -> None:
    pass

update_status("pending")    # ✅ OK
update_status("unknown")    # ❌ Type error

# 可以组合
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

# 现代写法：用类型别名语句（Python 3.12+）
type Status = Literal["pending", "processing", "completed", "failed"]
```

### 3. Protocol：结构化子类型（鸭子类型 + 类型检查）

```python
from typing import Protocol

# 定义协议（鸭子类型的类型化版本）
class Drawable(Protocol):
    def draw(self) -> None: ...

class Resizable(Protocol):
    def resize(self, width: int, height: int) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Rectangle:
    def draw(self) -> None:
        print("Drawing rectangle")
    def resize(self, width: int, height: int) -> None:
        print(f"Resizing to {width}x{height}")

def render(obj: Drawable) -> None:
    obj.draw()

render(Circle())      # ✅ OK（有 draw 方法）
render(Rectangle())   # ✅ OK（有 draw 方法）
# render("string")   # ❌ Type error（没有 draw 方法）

# 协议组合
class DrawableAndResizable(Drawable, Resizable, Protocol):
    pass

def smart_render(obj: DrawableAndResizable) -> None:
    obj.draw()
    obj.resize(100, 100)

smart_render(Rectangle())  # ✅ OK
smart_render(Circle())     # ❌ Type error（缺少 resize 方法）
```

#### Protocol 实现原理深度解析

**设计哲学：结构化子类型 vs 名义子类型**

```python
# 传统：名义子类型- 必须显式继承
class Animal:
    def speak(self) -> None: pass

class Dog(Animal):  # 必须显式继承
    def speak(self) -> None: pass

def make_sound(animal: Animal) -> None:
    animal.speak()

make_sound(Dog())  # ✅ OK - 显式继承

# Protocol：结构化子类型 - 基于方法签名匹配
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None: pass  # 不需要显式继承！

def render(obj: Drawable) -> None:
    obj.draw()

render(Circle())  # ✅ OK - 有 draw 方法即可
```

**运行时实现机制**

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# Protocol 的本质（简化版实现）
class Protocol:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._is_protocol = True  # 标记为 Protocol 类

# 运行时验证
print(Drawable.__bases__)           # (Protocol,)
print(hasattr(Drawable, '_is_protocol'))  # True
print(Drawable._is_protocol)        # True
```

**类型检查器的匹配算法**

```python
# 类型检查器使用结构化匹配（伪代码）
def is_protocol_compliant(cls, protocol):
    for attr_name in dir(protocol):
        if attr_name.startswith('_'):
            continue
        # 检查 cls 是否有相同签名的属性/方法
        if not hasattr(cls, attr_name):
            return False
        # 检查类型签名是否匹配
        if not is_signature_compatible(
            get_protocol_attr_signature(protocol, attr_name),
            get_class_attr_signature(cls, attr_name)
        ):
            return False
    return True

# 示例
class Drawable(Protocol):
    def draw(self) -> None: ...
    def color(self) -> str: ...

class Circle:
    def draw(self) -> None: ...    # ✅ 有 draw
    def color(self) -> str: ...    # ✅ 有 color

class Square:
    def draw(self) -> None: ...    # ✅ 有 draw
    # ❌ 缺少 color

# is_protocol_compliant(Circle, Drawable)    → True
# is_protocol_compliant(Square, Drawable)    → False
```

**@runtime_checkable 的工作机制**

```python
from typing import Protocol, runtime_checkable

# 默认：运行时不可检查
class Drawable(Protocol):
    def draw(self) -> None: ...

isinstance(Circle(), Drawable)  # ❌ TypeError
# 错误信息：Protocols are not runtime checkable

# 启用：添加 @runtime_checkable
@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None: ...

class Square:
    def paint(self) -> None: ...  # 方法名不匹配

# 现在可以运行时检查
isinstance(Circle(), Drawable)   # ✅ True
isinstance(Square(), Drawable)   # ✅ False
isinstance("string", Drawable)   # ✅ False

# runtime_checkable 原理（简化）
def runtime_checkable(cls):
    def __instancecheck__(protocol, instance):
        instance_cls = type(instance)
        return _is_structural_match(instance_cls, protocol)

    cls.__instancecheck__ = __instancecheck__
    cls._is_runtime_protocol = True
    return cls

def _is_structural_match(cls, protocol):
    for name in dir(protocol):
        if name.startswith('_'):
            continue
        if not hasattr(cls, name):
            return False
    return True
```

**Protocol vs ABC vs 类继承对比**

| 特性 | Protocol | ABC (abstractmethod) | 类继承 |
|------|----------|---------------------|--------|
| **类型系统** | 结构化 | 名义化 | 名义化 |
| **需要显式继承** | ❌ 否 | ✅ 是 | ✅ 是 |
| **运行时开销** | 极小 | 小 | 无 |
| **isinstance 检查** | 需要 @runtime_checkable | ✅ 原生支持 | ✅ 原生支持 |
| **多继承** | ✅ 轻松组合 | ⚠️ 复杂 | ⚠️ 复杂 |
| **鸭子类型** | ✅ 完美支持 | ❌ 不支持 | ❌ 不支持 |

**高级特性：泛型 Protocol**

```python
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)

class Container(Protocol[T_co]):
    """泛型 Protocol"""
    def get(self) -> T_co: ...

class IntBox:
    def get(self) -> int:
        return 42

class StrBox:
    def get(self) -> str:
        return "hello"

def use_container(c: Container[int]) -> int:
    return c.get()

use_container(IntBox())  # ✅ OK
# use_container(StrBox())  # ❌ Type error
```

**高级特性：只读与读写属性**

```python
from typing import Protocol

class ReadOnlyContainer(Protocol):
    """只读容器 Protocol"""
    @property
    def size(self) -> int: ...  # 只有 getter

class WriteableContainer(Protocol):
    """可写容器 Protocol"""
    @property
    def size(self) -> int: ...

    @size.setter
    def size(self, value: int) -> None: ...

class MyContainer:
    def __init__(self) -> None:
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = value

def read_only(c: ReadOnlyContainer) -> None:
    print(c.size)
    # c.size = 10  # ❌ Type error - Protocol 只有 getter
```

**实际应用：插件系统**

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Plugin(Protocol):
    """插件接口"""
    name: str
    version: str

    def load(self) -> None: ...
    def unload(self) -> None: ...

# 任何人都可以写插件，无需继承
class MyPlugin:
    name = "my_plugin"
    version = "1.0.0"

    def load(self) -> None:
        print("Plugin loaded")

    def unload(self) -> None:
        print("Plugin unloaded")

# 插件管理器
class PluginManager:
    def __init__(self) -> None:
        self.plugins: list[Plugin] = []

    def register(self, plugin: object) -> None:
        if isinstance(plugin, Plugin):
            self.plugins.append(plugin)
        else:
            raise TypeError("Not a valid plugin")

manager = PluginManager()
manager.register(MyPlugin())  # ✅ OK - 结构化匹配
```

**快速参考**

```python
# 基础 Protocol
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# 运行时可检查
from typing import runtime_checkable

@runtime_checkable
class Flyable(Protocol):
    def fly(self) -> None: ...

# Protocol 组合
class Amphibious(Flyable, Swimmable, Protocol):
    pass

# 泛型 Protocol
from typing import TypeVar

T = TypeVar("T")
class Container(Protocol[T]):
    def get(self) -> T: ...
```

### 4. TypedDict：结构化的字典类型

```python
from typing import TypedDict, Required, NotRequired

# Python 3.11+ 语法
class User(TypedDict):
    name: Required[str]           # 必需字段
    age: NotRequired[int]         # 可选字段
    email: str | None             # 可为 None

# 等价于（显式声明 total=False）
class UserOptional(TypedDict, total=False):
    name: str
    age: int
    email: str | None

def create_user(user: User) -> None:
    print(f"User: {user['name']}")

create_user({"name": "Alice"})           # ✅ OK
create_user({"name": "Bob", "age": 30})  # ✅ OK
create_user({"age": 30})                 # ❌ Type error（缺少 name）

# 配合 dataclass 使用
from dataclasses import dataclass

@dataclass
class UserModel:
    name: str
    age: int | None = None

def user_to_dict(user: UserModel) -> User:
    result: User = {"name": user.name}
    if user.age is not None:
        result["age"] = user.age
    return result
```

### 5. ParamSpec：用于装饰器的可变参数

```python
from typing import ParamSpec, Callable, TypeVar

P = ParamSpec('P')  # 捕获参数类型
R = TypeVar('R')    # 捕获返回类型

def decorator(func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@decorator
def add(a: int, b: int) -> int:
    return a + b

# 类型检查器知道 add 仍然是 (int, int) -> int
result: int = add(1, 2)  # ✅ 类型正确保留
```

### 6. Self：返回自身类型的方法

```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self

    def set_age(self, age: int) -> Self:
        self.age = age
        return self

# 链式调用类型安全
builder = Builder().set_name("Alice").set_age(30)  # 类型是 Builder
# 不用 Self 时，返回类型会是 Any 或需要手动指定泛型
```

### 7. TypeGuard：用户自定义类型谓词

```python
from typing import TypeGuard, Any

def is_string_list(val: list[Any]) -> TypeGuard[list[str]]:
    """检查是否为字符串列表，如果是，类型检查器会知道"""
    return all(isinstance(x, str) for x in val)

def process(items: list[Any]) -> None:
    if is_string_list(items):
        # 在这个分支，items 被视为 list[str]
        for item in items:
            print(item.upper())  # ✅ OK
    else:
        # 在这个分支，items 仍然是 list[Any]
        pass
```

### 8. Final：防止继承和重新赋值

```python
from typing import Final

# 常量声明
MAX_CONNECTIONS: Final = 100
# MAX_CONNECTIONS = 200  # ❌ Type error: cannot assign to Final

# 防止方法被覆盖
class Base:
    def method(self) -> None: ...

    def final_method(self) -> Final[None]:  # 子类不能覆盖
        pass

class Derived(Base):
    def method(self) -> None: ...        # ✅ OK

    # def final_method(self) -> None: ...  # ❌ Type error
```

## 四、常见的坑与注意事项

### 1. `from __future__ import annotations` 的双刃剑

```python
from __future__ import annotations  # Python 3.7+ 可用

# 好处：可以使用前向引用，无需引号
class Node:
    def __init__(self, value: int, next: Node | None = None):
        self.value = value
        self.next = next

# 坑：注解变成字符串，不是实际类型
def get_type() -> type:
    return list[str]  # ❌ 运行时错误！list[str] 不是合法的运行时表达式

# 正确做法：如果需要运行时类型，用 eval()
import typing
def get_type() -> type:
    return eval("list[str]", {"__builtins__": {}}, typing.__dict__)

# 或者不用 future import，用引号
class Node:
    def __init__(self, value: int, next: "Node | None" = None):
        pass
```

**建议**：
- **库代码**：使用 `from __future__ import annotations`（避免循环导入）
- **需要运行时反射的代码**：不使用，或谨慎处理

### 2. Any 是类型黑洞

```python
from typing import Any

def dangerous(x: Any) -> None:
    x.any_method()      # ✅ 通过类型检查（可能运行时错误）
    x + 1               # ✅ 通过类型检查
    whatever = x        # whatever 的类型是 Any

# 一旦接触 Any，类型会传播
def process(data: Any) -> int:
    result = data  # result 的类型是 Any，不是 int
    return result  # 类型检查器认为是 OK 的
```

**建议**：优先使用 `object` 或具体类型，只在必要时用 `Any`

### 3. 可变默认参数 + 类型注解

```python
from typing import DefaultDict

# ❌ 危险：可变默认参数
def foo(items: list[str] = []) -> None:  # 共享同一个列表对象！
    items.append("item")

# ✅ 正确：使用 None
def foo(items: list[str] | None = None) -> None:
    if items is None:
        items = []
    items.append("item")
```

### 4. 泛型方差陷阱

```python
from typing import TypeVar, Generic

T_co = TypeVar("T_co", covariant=True)   # 协变
T_contra = TypeVar("T_contra", contravariant=True)  # 逆变

class Box(Generic[T_co]):
    def __init__(self, value: T_co) -> None:
        self.value = value

    def get(self) -> T_co:  # ✅ 只读，可以是协变
        return self.value

    # ❌ 如果添加 set，就不能是协变了
    # def set(self, value: T_co) -> None:  # 类型错误

# 协变：Box[Derived] 可以赋值给 Box[Base]
class Animal: pass
class Dog(Animal): pass

class Producer(Generic[T_co]):
    def produce(self) -> T_co: ...

producer_dog: Producer[Dog] = Producer()
producer_animal: Producer[Animal] = producer_dog  # ✅ OK（协变）
```

### 5. TypedDict vs dataclass vs pydantic

```python
from typing import TypedDict
from dataclasses import dataclass
from pydantic import BaseModel, Field

# TypedDict：零运行时开销，纯类型注解
class UserDict(TypedDict):
    name: str
    age: int

user1: UserDict = {"name": "Alice", "age": 30}
# user1.name  # ❌ 运行时：user1 只是 dict，没有属性访问

# dataclass：轻量级，带代码生成
@dataclass
class UserData:
    name: str
    age: int

user2 = UserData("Alice", 30)
# user2.name  # ✅ 支持属性访问
# UserData(name="Bob", age="invalid")  # ❌ 运行时错误（但类型不检查）

# pydantic：运行时验证 + 类型注解
class UserModel(BaseModel):
    name: str
    age: int

user3 = UserModel(name="Alice", age=30)      # ✅ OK
# UserModel(name="Bob", age="invalid")      # ❌ 运行时验证错误

# **选择建议**：
# - 纯类型检查：TypedDict（FastAPI 请求/响应模型）
# - 简单数据容器：dataclass
# - 需要运行时验证：pydantic（FastAPI 实际上用 pydantic 做验证）
```

## 五、Linter 与类型检查工具

### 主流工具对比

| 工具 | 速度 | 严格度 | 生态系统 | 推荐场景 |
|------|------|--------|----------|----------|
| **mypy** | 中等 | 可配置 | 最成熟 | 通用项目 |
| **pyright** | 快 | 非常严格 | VS Code 原生支持 | 严格项目、VS Code 用户 |
| **pyre** | 快 | 严格 | Meta 开源 | 大型代码库 |
| **pytype** | 慢 | 推断型 | Google 开源 | 遗留代码添加类型 |

### 推荐配置

**pyproject.toml**（多工具配置）：

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "third_party_lib.*"
ignore_missing_imports = true

[tool.pyright]
include = ["src"]
exclude = ["**/node_modules", "**/__pycache__"]
typeCheckingMode = "strict"
reportMissingTypeStubs = false

[tool.ruff]
target-version = "py311"
select = ["ALL"]
ignore = ["D203", "D213"]  # 冲突的 docstring 规则
```

### 实用工具链

```bash
# 安装
pip install mypy pyright ruff pre-commit

# ruff：超快的 linter（替代 flake8 + isort + pydocstyle）
ruff check .
ruff check --fix .

# mypy：类型检查
mypy .

# pyright：更严格的类型检查
pyright .

# pre-commit：git hook 自动化
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]
EOF

pre-commit install
```

## 六、运行时行为与性能

### 类型注解的运行时开销

```python
import sys
from typing import NamedTuple, TypedDict
from dataclasses import dataclass

# 1. 类型注解存储在 __annotations__ 中（几乎无开销）
def func(x: int) -> str:
    return str(x)

print(func.__annotations__)  # {'x': <class 'int'>, 'return': <class 'str'>}

# 2. 不同数据结构的性能对比
import timeit

# TypedDict：零运行时开销（只是 dict）
class UserTD(TypedDict):
    name: str
    age: int

# dataclass：轻微开销（__init__ 等方法生成）
@dataclass
class UserDC:
    name: str
    age: int

# 普通类：无开销
class UserCls:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# 性能测试
N = 1_000_000
timeit.timeit(lambda: UserTD(name="Alice", age=30), number=N)  # 最快
timeit.timeit(lambda: UserDC("Alice", 30), number=N)          # 稍慢
timeit.timeit(lambda: UserCls("Alice", 30), number=N)         # 中等
```

### `@runtime_checkable`：运行时类型检查

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        pass

# 运行时 isinstance 检查
print(isinstance(Circle(), Drawable))  # ✅ True
print(isinstance("string", Drawable))  # ❌ False

# 不加 @runtime_checkable 会报错
# isinstance(Circle(), Drawable)  # ❌ TypeError: Protocols are not runtime checkable
```

### `cast()`：绕过类型检查器

```python
from typing import cast, Any

data: Any = json.loads('{"name": "Alice"}')

# 类型检查器不知道 data 的具体结构
# name = data["name"]  # name 的类型是 Any

# 使用 cast 告诉类型检查器
name: str = cast(str, data["name"])

# 危险：类型检查器信任你，但运行时可能错误
# bad: int = cast(int, data["name"])  # 类型检查器 OK，但运行时是 str
```

## 七、FastAPI/Uvicorn 类型支持

### FastAPI 如何利用类型系统

```python
from fastapi import FastAPI, Query, Path, Body, HTTPException
from typing import Literal, Annotated
from pydantic import BaseModel, Field, validator

app = FastAPI()

# 1. 路径参数类型自动转换和验证
@app.get("/users/{user_id}")
async def get_user(user_id: int) -> dict[str, str]:
    # FastAPI 自动：
    # - 解析 URL 路径
    # - 转换为 int
    # - 验证失败返回 422
    return {"user_id": str(user_id)}

# 2. 查询参数 + 默认值 + 验证
@app.get("/search")
async def search(
    q: Annotated[str, Query(min_length=3, max_length=50)] = "",
    page: int = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 10,
) -> dict[str, Any]:
    return {"query": q, "page": page, "size": size}

# 3. 请求体：Pydantic 模型
class UserCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    email: Annotated[str, Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")]
    age: Annotated[int, Field(ge=0, le=150)] = 0

    @validator("email")
    def email_lowercase(cls, v: str) -> str:
        return v.lower()

@app.post("/users")
async def create_user(user: UserCreate) -> UserCreate:
    # user 数据已验证，类型安全
    return user

# 4. 响应模型 + 状态码
class UserResponse(BaseModel):
    id: int
    name: str
    email: str

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user_v2(user: UserCreate) -> UserResponse:
    # FastAPI 自动序列化 UserResponse 为 JSON
    # 且过滤掉不在 response_model 中的字段
    return UserResponse(id=1, name=user.name, email=user.email)

# 5. 联合类型 + 字面值
@app.get("/items/{item_type}")
async def get_item(
    item_type: Literal["book", "movie", "song"]
) -> dict[str, str]:
    items = {
        "book": "The Great Gatsby",
        "movie": "Inception",
        "song": "Bohemian Rhapsody"
    }
    return {"type": item_type, "name": items[item_type]}

# 6. 嵌套模型
class Address(BaseModel):
    street: str
    city: str

class UserWithAddress(BaseModel):
    name: str
    address: Address  # 嵌套

@app.post("/users-with-address")
async def create_user_with_address(user: UserWithAddress) -> UserWithAddress:
    return user
```

### FastAPI + Pydantic V2 类型特性

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Self

# Pydantic V2 (FastAPI 0.100+)
class UserModel(BaseModel):
    name: str
    age: int
    email: str | None = None

    # 新的验证器语法
    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()

    # 模型级别的验证
    @model_validator(mode="after")
    def validate_age_email(self) -> Self:
        if self.age < 18 and self.email is None:
            raise ValueError("minors must provide email")
        return self

# FastAPI 自动生成 OpenAPI 文档
# 访问 /docs 查看 Swagger UI
# 访问 /redoc 查看 ReDoc
```

## 八、高级特殊用法

### 1. 泛型类深度解析

#### 基础概念与语法

泛型类是可以处理多种类型的类，同时保持类型安全。它允许你编写一次代码，然后使用不同的类型实例化。

```python
from typing import TypeVar, Generic

T = TypeVar("T")  # 定义类型变量

class Box(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value

    def set(self, value: T) -> None:
        self.value = value

# 使用时指定具体类型
int_box: Box[int] = Box(42)
str_box: Box[str] = Box("hello")

int_box.set(100)       # ✅ OK
int_box.set("string")  # ❌ Type error: expected int, got str

# 类型检查器知道返回类型
x: int = int_box.get()  # ✅ OK
```

#### 多个类型变量

```python
from typing import TypeVar, Generic

K = TypeVar("K")  # Key 类型
V = TypeVar("V")  # Value 类型

class KeyValuePair(Generic[K, V]):
    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value

    def get_key(self) -> K:
        return self.key

    def get_value(self) -> V:
        return self.value

# 使用
pair1: KeyValuePair[str, int] = KeyValuePair("age", 30)
pair2: KeyValuePair[int, str] = KeyValuePair(1, "one")

key: str = pair1.get_key()   # ✅ "age"
value: int = pair1.get_value()  # ✅ 30
```

#### TypeVar 深度解析

**bound：类型约束**

```python
from typing import TypeVar

class Animal:
    def speak(self) -> str:
        return "Some sound"

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

# T 必须是 Animal 或其子类
T_Animal = TypeVar("T_Animal", bound=Animal)

def make_speak(animal: T_Animal) -> str:
    return animal.speak()

make_speak(Dog())  # ✅ 返回 "Woof!"
# make_speak("string")  # ❌ Type error
```

**协变与逆变**

```python
from typing import TypeVar, Generic

class Animal: pass
class Dog(Animal): pass

# 协变（只读）
T_co = TypeVar("T_co", covariant=True)

class ReadOnlyBox(Generic[T_co]):
    def get(self) -> T_co: ...

dog_box = ReadOnlyBox(Dog())
animal_box: ReadOnlyBox[Animal] = dog_box  # ✅ OK（协变）

# 逆变（只写）
T_contra = TypeVar("T_contra", contravariant=True)

class WriteOnlyBox(Generic[T_contra]):
    def set(self, value: T_contra) -> None: ...

animal_writer = WriteOnlyBox[Animal]()
dog_writer: WriteOnlyBox[Dog] = animal_writer  # ✅ OK（逆变）
```

#### 实际应用场景

**1. 容器类型**

```python
from typing import TypeVar, Generic, Iterator

T = TypeVar("T")

class Stack(Generic[T]):
    """类型安全的栈"""
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

# 使用
int_stack = Stack[int]()
int_stack.push(1)
int_stack.push("three")  # ❌ Type error
```

**2. 结果类型（Result / Either 模式）**

```python
from typing import TypeVar, Generic

Ok = TypeVar("Ok")
Err = TypeVar("Err")

class Result(Generic[Ok, Err]):
    """表示可能失败的操作结果"""
    def __init__(self, value: Ok | Err, is_ok: bool) -> None:
        self._value = value
        self._is_ok = is_ok

    @staticmethod
    def ok(value: Ok) -> "Result[Ok, Err]":
        return Result(value, True)

    def is_ok(self) -> bool:
        return self._is_ok

# 使用
def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Result.err("Division by zero")
    return Result.ok(a / b)
```

**3. 构建器模式**

```python
from typing import TypeVar, Generic

T = TypeVar("T", bound="Builder")

class Builder(Generic[T]):
    """支持链式调用的构建器基类"""
    def set_option(self, key: str, value: object) -> T:
        return self  # type: ignore

class ServerBuilder(Builder["ServerBuilder"]):
    def set_host(self, host: str) -> "ServerBuilder":
        return self.set_option("host", host)  # type: ignore

# 使用：链式调用类型安全
server = ServerBuilder().set_host("localhost").set_port(8080)
```

#### 常见陷阱

**1. 类型擦除**

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

# ⚠️ 运行时类型参数会被擦除
box = Box(42)
print(type(box))  # <class '__main__.Box'> - 不是 Box[int]
```

**2. 继承泛型类**

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Base(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

# ❌ 错误：没有指定类型变量
# class DerivedWrong(Base): pass

# ✅ 正确：固定类型
class IntBox(Base[int]):
    pass

int_box = IntBox(42)
# int_box = IntBox("string")  # ❌ Type error

# ✅ 正确：保持泛型
class Box(Base[T]):
    pass
```

#### 快速参考总结

```python
# 基础泛型类
from typing import TypeVar, Generic
T = TypeVar("T")
class Box(Generic[T]): ...

# 多个类型变量
K = TypeVar("K")
V = TypeVar("V")
class Map(Generic[K, V]): ...

# bound 约束
T = TypeVar("T", bound=BaseClass)
class Container(Generic[T]): ...

# 协变（只读）
T_co = TypeVar("T_co", covariant=True)
class ReadOnly(Generic[T_co]): ...

# 逆变（只写）
T_contra = TypeVar("T_contra", contravariant=True)
class WriteOnly(Generic[T_contra]): ...
```

### 2. 类型守卫

```python
from typing import Any, TypeGuard

def is_list_of_strings(val: Any) -> TypeGuard[list[str]]:
    return isinstance(val, list) and all(isinstance(x, str) for x in val)

def process(data: Any) -> None:
    if is_list_of_strings(data):
        # 这里 data 被认为是 list[str]
        print([s.upper() for s in data])
```

### 3. 递归类型

```python
from typing import List, Union

# JSON 类型定义（递归）
JsonType = Union[None, bool, int, float, str, List["JsonType"], dict[str, "JsonType"]]

def parse_json(data: str) -> JsonType:
    import json
    return json.loads(data)

# Python 3.11+ 使用 type 语句更简洁
type JsonType2 = (
    None | bool | int | float | str | list[JsonType2] | dict[str, JsonType2]
)
```

### 4. Callable 类型

```python
from typing import Callable

# 简单函数类型
def apply(func: Callable[[int, int], int], x: int, y: int) -> int:
    return func(x, y)

def add(a: int, b: int) -> int:
    return a + b

apply(add, 1, 2)  # ✅ OK
# apply("not a function", 1, 2)  # ❌ Type error

# 更灵活的 Callable
def register_callback(callback: Callable[..., None]) -> None:
    callback()

register_callback(lambda: print("hello"))
register_callback(lambda x: print(x))  # ✅ OK（接受任意参数）
```

### 5. `@overload` 重载

```python
from typing import overload, Union

@overload
def process(data: str) -> str: ...

@overload
def process(data: int) -> str: ...

@overload
def process(data: list[str]) -> list[str]: ...

def process(data: Union[str, int, list[str]]) -> Union[str, list[str]]:
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, int):
        return str(data)
    else:
        return [s.upper() for s in data]

# 类型检查器根据参数类型推断返回类型
result1: str = process("hello")           # 返回类型是 str
result2: str = process(123)               # 返回类型是 str
result3: list[str] = process(["a", "b"])  # 返回类型是 list[str]
```

### 6. `NoReturn` 标记永不返回的函数

```python
from typing import NoReturn

def fail() -> NoReturn:
    raise RuntimeError("Something went wrong")

def always_fail() -> NoReturn:
    fail()
    # 这行永远不会执行
    print("unreachable")  # 类型检查器会警告
```

### 7. `LiteralString` 防止 SQL 注入

```python
from typing import LiteralString

def execute_query(query: LiteralString) -> None:
    # 只有字面字符串可以传入，不能是用户输入
    pass

execute_query("SELECT * FROM users")  # ✅ OK
user_input = "SELECT * FROM users"
# execute_query(user_input)  # ❌ Type error（不是字面字符串）
```

## 九、快速参考：常用类型速查表

```python
# 基础类型
x: int
x: float
x: str
x: bool
x: bytes

# 容器类型（Python 3.9+）
x: list[str]
x: dict[str, int]
x: set[str]
x: tuple[str, int]
x: tuple[str, ...]  # 可变长度

# 可选与联合（Python 3.10+）
x: str | None       # 可选
x: int | str        # 联合
x: int | str | None # 联合 + 可选

# 常用 typing 类型
from typing import Optional, Union, Any, NoReturn, Never
x: Optional[str]    # 等同于 str | None
x: Union[int, str]  # 等同于 int | str
x: Any              # 任意类型
def f() -> NoReturn: ...  # 永不返回
def g() -> Never: ...     # 永不返回（3.11+）

# 特殊类型
from typing import Literal, Final, TypeAlias
Status: TypeAlias = Literal["pending", "done"]
MAX_SIZE: Final = 100

# 函数类型
from typing import Callable
x: Callable[[int, str], bool]  # (int, str) -> bool

# 泛型
from typing import TypeVar, Generic
T = TypeVar("T")
class Box(Generic[T]): ...

# Protocol
from typing import Protocol
class Drawable(Protocol):
    def draw(self) -> None: ...

# TypedDict
from typing import TypedDict, Required, NotRequired
class User(TypedDict):
    name: Required[str]
    age: NotRequired[int]

# 类型守卫
from typing import TypeGuard
def is_str(val: object) -> TypeGuard[str]:
    return isinstance(val, str)
```

## 学习笔记

1. **类型系统是渐进式的**：不需要一次性给所有代码加类型，可以从关键模块开始
2. **工具选择**：mypy 成熟稳定，pyright 更严格更快（VS Code 原生支持）
3. **FastAPI 深度集成**：类型注解不仅用于检查，还用于自动生成文档、验证、序列化
4. **运行时 vs 静态**：Python 类型主要是静态检查工具使用，运行时基本忽略
5. **性能**：类型注解本身几乎没有运行时开销，dataclass/TypedDict 等有轻微开销

## 后续行动计划

1. 在实际项目中尝试使用 strict 模式的 mypy/pyright
2. 学习 FastAPI 的依赖注入系统和类型验证机制
3. 了解 Pydantic V2 的新特性和性能优化
4. 实践编写复杂的泛型类和装饰器
5. 探索类型推导和类型推断的边界情况

## 参考资料

- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 604 - Allow writing union types as X | Y](https://peps.python.org/pep-0604/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Pydantic V2 documentation](https://docs.pydantic.dev/)
