"""权重手术：让「输入维度变化后仍能热启动」成为可能。

**为什么需要**
`torch.load` 要求状态字典形状严格匹配。所以只要改了观测/候选维度或网络宽度，
`--init-from` 就会报 size mismatch，过去只能**从零重训**（V4→V5→V6→V7→V9 每次都这样）。

但绝大多数维度变化其实是「旧维度的一个超集或重排」，可以**手术式改写第一层权重**，
把已有知识完整搬过去，再继续训练：

    加新输入维度  → 新增列置 0（新特征初始不影响输出，模型从"等价于旧模型"起步）
    删输入维度    → 切掉对应列
    重排输入维度  → 按名字/索引重映射列（关键：语义变了绝不能只靠位置对齐）
    加宽隐藏层    → 权重矩阵块对角填充，新神经元零初始化（梯度仍会流，会自己学起来）

**最重要的一条**：如果维度变化的本质是**语义重排**（例如观测药水槽位换了顺序），
只按位置对齐会**注入错误的权重**，模型会以错误先验起步——比从零重训更糟。
这种情况必须传 remap（见 `name_remap`）。

**可验证的性质**（见 tests/test_weight_surgery.py）：
  用「零填充新增列」的方式扩展维度后，在新增维全为 0 的输入上，
  手术后模型的输出与手术前**逐元素相同**。这不是近似，是可断言的恒等性。
"""

from typing import List, Optional, Sequence

import torch


def identity_map(n: int) -> List[Optional[int]]:
    """长度 n 的恒等映射（新位置 i ← 旧位置 i）。"""
    return list(range(n))


def append_map(old_n: int, new_n: int) -> List[Optional[int]]:
    """保留旧的前 old_n 维，新增 (new_n - old_n) 维置 None（零填充）。"""
    assert new_n >= old_n, f"只支持扩张：old={old_n} new={new_n}"
    return list(range(old_n)) + [None] * (new_n - old_n)


def name_remap(old_names: Sequence[str], new_names: Sequence[str]) -> List[Optional[int]]:
    """按名字把新位置映射回旧位置；新名字在旧表里找不到 → None（零填充）。

    这是处理「语义重排」的正确方式：例如观测里药水 one-hot 槽位换了顺序，
    必须按药水名对齐，不能按下标对齐。
    """
    pos = {n: i for i, n in enumerate(old_names)}
    return [pos.get(n) for n in new_names]


# --------------------------------------------------------------------------
# 观测段的分段布局：把「维度变化」翻译成正确的输入映射
# --------------------------------------------------------------------------
# 旧（468）：[0,412) C++ 原始 ｜ [412,436) 药水 one-hot 24 ｜ [436,437) 药水数
#           ｜ [437,441) 地图摘要 ｜ [441,445) Act ｜ [445,455) 房间
#           ｜ [455,460) 牌组流派 ｜ [460,468) 牌组派生统计
# 新（477）：[0,412) 同 ｜ [412,445) 药水 one-hot 33 ｜ [445,446) 药水数
#           ｜ [446,450) 地图 ｜ [450,454) Act ｜ [454,464) 房间
#           ｜ [464,469) 流派 ｜ [469,477) 派生统计
#
# ⚠️ 关键教训：药水段 24→33 不只是「变多」，而是**槽位顺序整体换了**
#    （旧表是跨角色常见 24 种，新表是铁甲药水池顺序）。所以必须按名字对齐；
#    而药水段**之后**的所有段则是整体后移 9 维，必须按位置 -9 对齐。
#    只按位置对齐会静默注入**错误先验**——比从零重训更糟。
_OBS_OLD_TAIL = 468          # 旧观测总维
_OBS_NEW_TAIL = 477          # 新观测总维
_OBS_POTION_OFF = 412        # 药水段起点（新旧相同）
_OBS_OLD_POTION_N = 24
_OBS_NEW_POTION_N = 33

# 旧 24 槽药水顺序（跨角色常见 24 种；含 3 个铁甲拿不到的异角色药水）
POTION_OLD_24 = [
    "ATTACK_POTION", "BLOCK_POTION", "BLOOD_POTION", "STRENGTH_POTION",
    "DEXTERITY_POTION", "ENERGY_POTION", "FIRE_POTION", "POISON_POTION",
    "WEAK_POTION", "SPEED_POTION", "FLEX_POTION", "FOCUS_POTION",
    "SKILL_POTION", "POWER_POTION", "REGEN_POTION", "SWIFT_POTION",
    "EXPLOSIVE_POTION", "CUNNING_POTION", "DUPLICATION_POTION",
    "FRUIT_JUICE", "LIQUID_MEMORIES", "LIQUID_BRONZE", "SMOKE_BOMB",
    "ANCIENT_POTION",
]


def obs_468_to_477_map() -> List[Optional[int]]:
    """旧观测 468 维 → 新观测 477 维的**正确**输入映射。

    构成：原始段按位置不变 ｜ 药水段按名字重排 ｜ 后续 32 维整体后移 9。
    新出现的 12 个药水槽位映射为 None（零填充，初始不影响输出）。
    """
    from .obs_encoder import POTION_INDEX as new33

    m: List[Optional[int]] = list(range(_OBS_POTION_OFF))          # C++ 原始段
    for src in name_remap(POTION_OLD_24, new33):                    # 药水段
        m.append(None if src is None else _OBS_POTION_OFF + src)
    shift = _OBS_NEW_POTION_N - _OBS_OLD_POTION_N                    # 9
    tail_start = _OBS_POTION_OFF + _OBS_NEW_POTION_N                 # 445
    for i in range(tail_start, _OBS_NEW_TAIL):                       # 后续段整体后移
        m.append(i - shift)
    assert len(m) == _OBS_NEW_TAIL, len(m)
    # 映射源必须落在旧观测范围内或是 None
    assert all(x is None or 0 <= x < _OBS_OLD_TAIL for x in m)
    return m


def _adapt_linear(W: torch.Tensor, b: torch.Tensor,
                  in_map: Optional[List[Optional[int]]],
                  out_map: Optional[List[Optional[int]]]):
    """按映射重建一个 Linear 层的 (weight, bias)。

    W: (out_old, in_old)；in_map 长度 = 新输入维度；out_map 长度 = 新输出维度。
    None 表示该位置是新维度 → 取 0（零初始化）。
    """
    out_new = len(out_map) if out_map is not None else W.shape[0]
    in_new = len(in_map) if in_map is not None else W.shape[1]

    new_W = torch.zeros(out_new, in_new, dtype=W.dtype)
    new_b = torch.zeros(out_new, dtype=b.dtype)

    rows = range(out_new)
    for r_new in rows:
        r_old = out_map[r_new] if out_map is not None else (
            r_new if r_new < W.shape[0] else None)
        if r_old is None:
            continue
        if b is not None:
            new_b[r_new] = b[r_old]
        for c_new in range(in_new):
            c_old = in_map[c_new] if in_map is not None else (
                c_new if c_new < W.shape[1] else None)
            if c_old is None:
                continue
            new_W[r_new, c_new] = W[r_old, c_old]
    return new_W, new_b


def _linear_keys(sd: dict, prefix: str) -> List[str]:
    """按顺序取出 prefix 下所有 Linear 的 weight 键（如 mlp.0.weight, mlp.2.weight）。"""
    idx = []
    for k in sd:
        if k.startswith(prefix) and k.endswith(".weight"):
            mid = k[len(prefix):-len(".weight")]
            if "." in mid:
                head, _, tail = mid.rpartition(".")
                if head == "" and tail.isdigit():
                    idx.append((int(tail), k))
            elif mid.isdigit():
                idx.append((int(mid), k))
    idx.sort()
    return [k for _, k in idx]


def adapt_mlp_state(sd: dict, *, prefix: str,
                    old_in: int, new_in: int,
                    old_hidden: Sequence[int], new_hidden: Sequence[int],
                    in_map: Optional[List[Optional[int]]] = None) -> dict:
    """把以 `prefix` 命名的 MLP 状态字典适配到新的输入维度/隐藏层宽度。

    只支持扩张（new ≥ old）。in_map 长度必须等于 new_in。
    """
    assert list(new_hidden) == list(new_hidden), "new_hidden 必须可比较"
    new_hidden = list(new_hidden)
    old_hidden = list(old_hidden)
    assert len(new_hidden) == len(old_hidden), "层数必须一致（本工具不做加层）"
    assert new_in >= old_in or in_map is not None, "缩维必须显式给 in_map"
    if in_map is None:
        assert new_in >= old_in, f"扩张才可不给 in_map：old={old_in} new={new_in}"
        in_map = append_map(old_in, new_in)
    assert len(in_map) == new_in, f"in_map 长度 {len(in_map)} != new_in {new_in}"

    out = dict(sd)
    keys = _linear_keys(sd, prefix)
    assert len(keys) == len(old_hidden) + 1, (
        f"{prefix} 的 Linear 层数 {len(keys)} != len(hidden)+1 {len(old_hidden)+1}")

    cur_in_map = in_map
    for li, wk in enumerate(keys):
        bk = wk[:-len("weight")] + "bias"
        W, b = sd[wk], sd[bk]
        is_last = (li == len(keys) - 1)
        if is_last:
            out_map = None                      # 输出恒为 1 维
        else:
            out_map = append_map(old_hidden[li], new_hidden[li])
        new_W, new_b = _adapt_linear(W, b, cur_in_map, out_map)
        out[wk], out[bk] = new_W, new_b
        # 下一层的输入 = 本层的输出：前 old_hidden[li] 个是旧神经元，其余置零。
        # 最后一层没有下一层，不再更新（否则会下标越界）。
        if not is_last:
            cur_in_map = append_map(old_hidden[li], new_hidden[li])
    return out


def adapt_checkpoint(src_path: str, dst_path: str, *, prefix: str,
                     old_in: int, new_in: int,
                     old_hidden: Sequence[int], new_hidden: Sequence[int],
                     in_map: Optional[List[Optional[int]]] = None) -> str:
    """读一个 checkpoint → 手术 → 写到 dst_path。返回 dst_path。"""
    sd = torch.load(src_path, weights_only=True)
    new_sd = adapt_mlp_state(sd, prefix=prefix, old_in=old_in, new_in=new_in,
                             old_hidden=old_hidden, new_hidden=new_hidden,
                             in_map=in_map)
    torch.save(new_sd, dst_path)
    return dst_path


def main(argv=None):
    """CLI：给一个 Scorer/ValueNet checkpoint 做维度/宽度手术。

    例（把 468 观测 + 443 候选的 V6 权重适配到 477 + 347）：
      python -m models.weight_surgery --src checkpoints/ppo_v6_5000.pt \\
        --dst checkpoints/ppo_v6_adapted.pt --prefix mlp --in-kind scorer \\
        --old-obs 468 --new-obs 477 --obs-recipe potion_24to33 \\
        --old-cand 443 --new-cand 347 --cand-drop 271,367 --hidden 512,256

    对齐规则（**错了会静默注入错误先验，比从零重训更糟**，故强制显式）：
      · 观测段不变            → 按位置恒等
      · 观测新维追加在末尾    → 需显式 --obs-append-only
      · 观测段语义重排        → 需 --obs-recipe（如药水 24→33）
      · 观测段其他变化        → CLI 拒绝执行，请构造 in_map 调 adapt_mlp_state()
      · 候选段                → 按 --cand-drop 的旧下标区间**按序删除**后对齐
                                （443→347 即删掉旧的 [271,367) 那段 96 维 label 哈希）
    """
    import argparse
    p = argparse.ArgumentParser(prog="models.weight_surgery")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--prefix", default="mlp")
    p.add_argument("--in-kind", choices=["scorer", "value"], default="scorer",
                   help="scorer = 输入是 obs⊕cand 拼接；value = 只有 obs")
    p.add_argument("--old-obs", type=int, required=True)
    p.add_argument("--new-obs", type=int, required=True)
    p.add_argument("--old-cand", type=int, default=0)
    p.add_argument("--new-cand", type=int, default=0)
    p.add_argument("--cand-drop", default=None,
                   help="按序删除的旧候选下标区间，如 271,367（左闭右开）")
    p.add_argument("--obs-recipe", default=None, choices=["potion_24to33"],
                   help="观测段变化的对齐方案；不确定就别填（会强制报错而不是猜）")
    p.add_argument("--obs-append-only", action="store_true",
                   help="明确声明：观测新维度是【追加在末尾】的")
    p.add_argument("--hidden", required=True, help="逗号分隔，如 512,256")
    p.add_argument("--new-hidden", default=None,
                   help="不填则与 --hidden 相同（只改输入维度）")
    args = p.parse_args(argv)

    # ---- 构造输入段映射（长度必须等于 new 输入维度）----
    if args.new_obs == args.old_obs:
        obs_map: List[Optional[int]] = list(range(args.old_obs))
    elif args.obs_recipe == "potion_24to33":
        assert (args.old_obs, args.new_obs) == (468, 477), \
            "--obs-recipe potion_24to33 只适用于 468→477"
        obs_map = obs_468_to_477_map()
    elif args.obs_append_only:
        obs_map = append_map(args.old_obs, args.new_obs)
    else:
        raise SystemExit(
            f"观测维度 {args.old_obs} → {args.new_obs} 必须显式说明对齐方式：\n"
            f"  · 新维度确实【追加在末尾】        → --obs-append-only\n"
            f"  · 药水段 24→33（重排+后续段后移） → --obs-recipe potion_24to33\n"
            f"  · 其他情况                        → 在 Python 里构造 in_map "
            f"调 adapt_mlp_state()\n"
            f"按位置硬对齐会静默注入【错误先验】，比从零重训更糟，故强制显式说明。")

    in_map: List[Optional[int]] = list(obs_map)
    if args.in_kind == "scorer":
        if args.cand_drop:
            a, b = (int(x) for x in args.cand_drop.split(","))
        else:
            a, b = None, None
        kept = [i for i in range(args.old_cand)
                if a is None or not (a <= i < b)]
        if len(kept) != args.new_cand and len(kept) < args.new_cand:
            kept = kept + [None] * (args.new_cand - len(kept))   # 允许末尾追加新维
        assert len(kept) >= args.new_cand, (
            f"旧候选维去掉 --cand-drop 后只剩 {len(kept)} 维，"
            f"不足 new_cand={args.new_cand}")
        for j in range(args.new_cand):
            src = kept[j]
            in_map.append(args.old_obs + src if src is not None else None)

    old_in = args.old_obs + (args.old_cand if args.in_kind == "scorer" else 0)
    new_in = args.new_obs + (args.new_cand if args.in_kind == "scorer" else 0)
    assert len(in_map) == new_in, f"in_map {len(in_map)} != new_in {new_in}"

    hidden = tuple(int(x) for x in args.hidden.split(","))
    new_hidden = (tuple(int(x) for x in args.new_hidden.split(","))
                  if args.new_hidden else hidden)
    adapt_checkpoint(args.src, args.dst, prefix=args.prefix,
                     old_in=old_in, new_in=new_in,
                     old_hidden=hidden, new_hidden=new_hidden, in_map=in_map)
    n_new = sum(1 for x in in_map if x is None)
    print(f"[surgery] {args.src} → {args.dst}")
    print(f"[surgery] 输入 {old_in} → {new_in}；hidden {hidden} → {new_hidden}")
    print(f"[surgery] 其中 {n_new} 个新输入维置 0（初始不影响输出，之后自行学习）")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
