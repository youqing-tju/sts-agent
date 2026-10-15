# -*- coding: utf-8 -*-
"""验证 CardId.value（整数）与 CardId.name（字符串）**一一对应**。

为什么必须验：fingerprint 原用 `c.id.name`（字符串，444us/张），拟换成
`c.id.value`（整数，1.24us/张）。若 value 有重复，则去重会把**不同的卡**判成同态
→ Beam 搜索语义被破坏。**快不是理由，等价才是。**
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts  # noqa: E402


def main():
    names = [a for a in dir(sts.CardId) if a.isupper() and not a.startswith("_")]
    pairs = []
    bad = []
    for n in names:
        try:
            v = getattr(sts.CardId, n)
        except Exception as e:                       # noqa: BLE001
            bad.append((n, "getattr->%s" % e))
            continue
        if not isinstance(v, sts.CardId):
            bad.append((n, "not CardId: %r" % (type(v),)))
            continue
        try:
            pairs.append((int(v.value), v.name, n))
        except Exception as e:                       # noqa: BLE001
            bad.append((n, "value->%s" % e))

    print("枚举到的 CardId 常量数：", len(pairs), " 取不到的：", len(bad))
    if bad[:5]:
        print("  取不到的样例：", bad[:5])

    # ① value 唯一？
    by_val: dict[int, list[str]] = {}
    for val, nm, attr in pairs:
        by_val.setdefault(val, []).append(nm)
    dup_val = {v: ns for v, ns in by_val.items() if len(ns) > 1}
    print("\n① value 唯一性：", "✅ 全部唯一" if not dup_val else "🔴 重复 %d 个" % len(dup_val))
    if dup_val:
        for v, ns in list(dup_val.items())[:10]:
            print("     value=%s -> %s" % (v, ns))

    # ② name 唯一？
    by_name: dict[str, list[int]] = {}
    for val, nm, attr in pairs:
        by_name.setdefault(nm, []).append(val)
    dup_name = {n: vs for n, vs in by_name.items() if len(vs) > 1}
    print("② name  唯一性：", "✅ 全部唯一" if not dup_name else "⚠️ 重复 %d 个" % len(dup_name))

    # ③ 属性名（如 STRIKE_RED）与 name 是否一致（大小写差异只作备注）
    mismatch = [(attr, nm) for val, nm, attr in pairs if attr != nm]
    print("③ 属性名 vs .name 不一致：", len(mismatch), "个", mismatch[:5] if mismatch else "")

    # ④ 双向映射是否 1:1
    ok = (not dup_val) and (not dup_name) and (len(by_val) == len(pairs))
    print("\n结论：", "✅ value ↔ name 是一一映射，可安全替换" if ok else "🔴 不可直接替换")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
