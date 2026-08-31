"""对局日志包。

注意：本包名与标准库 logging 同名。为了让 pytest 等依赖标准库 logging 的
第三方库在项目根目录下正常工作，这里把标准库 logging 的全部内容加载进本
命名空间，并把标准库 logging 目录并入 __path__（使 logging.config /
logging.handlers 等子模块仍可导入），然后再追加本项目的 RunLogger。
"""
import os as _os

_stdlib_logging_dir = _os.path.join(_os.path.dirname(_os.__file__), "logging")

# 子模块解析路径：先本包目录（run_logger 等），后标准库 logging 目录（config/handlers）
__path__ = [_os.path.dirname(__file__), _stdlib_logging_dir]

# 在当前命名空间执行标准库 logging/__init__.py，等价于 `from logging import *` 的完整版
_stdlib_init = _os.path.join(_stdlib_logging_dir, "__init__.py")
with open(_stdlib_init, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _stdlib_init, "exec"), globals())

from .run_logger import RunLogger

__all__ = [name for name in dir() if not name.startswith("_")]
