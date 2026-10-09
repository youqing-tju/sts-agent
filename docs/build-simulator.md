# 编译 C++ 模拟器（sts_lightspeed → slaythespire.pyd）

> **多数情况下你不需要编译。**
> 仓库已附带编译好的 `slaythespire.cp313-win_amd64.pyd`
> （CPython 3.13 / Windows x64，2026-09-14 构建）。
> 建好 `.venv` 装完 `requirements.txt` 就能直接跑。
> 只有以下两种情况才需要自己编译：
> ① 目标平台不是 Windows x64 或 Python 不是 3.13（`.pyd` 的 ABI 标签是 `cp313`）
> ② 修改了 `third_party/sts_lightspeed/` 下的 C++ 源码

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 x64 |
| **Visual Studio** | **2022**（Professional / Enterprise / Community / **Build Tools** 任一版本均可） |
| VS 工作负载 | 必须勾选 **「使用 C++ 的桌面开发」**（提供 MSVC v143 工具集 + Windows SDK + CMake/Ninja 组件） |
| Python | **3.13**（其他版本产出的 pyd 标签不同，无法 import；此时需按你的版本重新编译） |
| 项目环境 | 项目根已执行 `python -m venv .venv` 且 `pip install -r requirements.txt` |

> ⚠️ 只装 VS 本体但不勾选 C++ 工作负载 → 找不到 `cl.exe`，编译会在第一步失败。

## 2. 编译

在**项目根目录**执行：

```bat
third_party\sts_lightspeed\build_bindings.bat
```

脚本会：探测 VS 安装位置 → 组装编译环境（不依赖 `VsDevCmd`，因为它需要 `reg.exe`）→
CMake + Ninja 构建 → **把产物自动复制到项目根**。

脚本头部有「用户配置区」（`VSPATH` / `SDKV` / `PYEXE` 三个变量）——
自动探测失败时（例如 VS 装在非标准路径），手动填这三个值即可。

## 3. 验证

```bat
.venv\Scripts\python.exe verify_env.py 12345
```

输出里应能看到模拟器版本与一次成功的环境自检。
更完整一点：`KMP_DUPLICATE_LIB_OK=TRUE .venv\Scripts\python.exe -m pytest tests\ -q`
（应当全绿；`.pyd` 不可用时会大面积 import 失败）。

## 4. 故障排除

| 症状 | 原因 / 处理 |
|---|---|
| `ImportError: DLL load failed` | Python 版本与 pyd 的 `cpXXX` 标签不匹配 → 用 Python 3.13，或自行重编译 |
| `ModuleNotFoundError: No module named 'slaythespire'` | pyd 不在**项目根**（不是 `third_party/` 下）→ 把构建产物复制到根目录 |
| 找不到 `cl.exe` | VS 未勾选「使用 C++ 的桌面开发」工作负载 |
| CMake 找不到 Python | `.venv` 未创建，或脚本的 `PYEXE` 指向错误 |
| 链接期大量 undefined symbol | Python 版本架构不符（用 x64 Python，别用 x86） |
| `ninja: error: loading 'build.ninja'` | 删掉 `third_party/sts_lightspeed/build/` 后重跑（缓存失效） |

## 5. 本仓库的构建环境（供参考）

原构建使用：Visual Studio 2022 Professional + MSVC 工具集 `14.42.34433` +
Windows SDK `10.0.22621.0` + Ninja + CMake（随 VS 提供）。
`build_bindings.bat` 原本把这些路径**写死为绝对路径**，已改为自动探测 + 可手动覆盖。

> 💡 **该项目当前维护者的本机已卸载 Visual Studio**，因此**无法重新编译**——
> 仓库内附带的 `slaythespire.cp313-win_amd64.pyd` 是唯一可用的模拟器二进制。
> 若需重建，先重装 VS 2022 并勾选 C++ 工作负载。
