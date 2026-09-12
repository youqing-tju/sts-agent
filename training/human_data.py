"""人类 .run → BC 训练样本生成器（V10 后训练/SFT 预热数据）。

流程：`.run` → 决策点 → 状态重建（模板帧 + 覆盖字段）→ 灌入模拟器
      → (obs 477, 候选 N×369, 标签 idx)

第一批聚焦 **CARD_REWARD 屏**——它承载用户最关心的两个能力（选牌偏好 / 跳牌），
且候选不依赖地图，可先用最可靠的方式打通。其余屏（EVENT/SHOP/REST/MAP）后续扩展。

用法：
    .venv/Scripts/python.exe -m training.human_data --out data/human_bc/card_reward.npz
    .venv/Scripts/python.exe -m training.human_data --max-runs 5 --dry-run   # 小规模自检
"""

import argparse
import copy
import glob
import json
import os
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RUN_GLOB = "data/external/Slay-the-Spire-data/runs/*/IRONCLAD/*.run"
TPL_PATH = "data/human_bc/_tpl_card_reward.json"
MAX_CANDS = 5          # 4 张卡 + SKIP 的 padding 上限

# 铁甲基础牌组（A0/A20 相同；A10+ 另有 AscendersBane 诅咒，由 master_deck 反推自然带入）
_BASE_DECK = {"Strike_R": 5, "Defend_R": 4, "Bash": 1}


# ---------------------------------------------------------------- 模板帧

def get_template():
    """取一个真机 CARD_REWARD 帧的 game_state 作为状态模板（缓存到磁盘）。

    为什么不从零组装 JSON：`init_from_json` 对 null / 缺字段极敏感
    （实测 `{}` 直接报 "type must be number, but is null"）。
    用真机帧当模板 + 覆盖字段是最稳的路径。
    """
    if os.path.exists(TPL_PATH):
        return json.load(open(TPL_PATH, encoding="utf-8"))
    tpl = None
    for L in sorted(glob.glob(os.path.join(_ROOT, "logs/realtime/*_raw.jsonl"))):
        try:
            for line in open(L, encoding="utf-8"):
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                gs = o.get("game_state") or {}
                if (gs.get("screen_type") == "CARD_REWARD" and gs.get("deck")
                        and len(((gs.get("screen_state") or {}).get("cards") or [])) == 3):
                    tpl = gs
                    break
        except FileNotFoundError:
            pass
        if tpl:
            break
    if tpl is None:
        raise RuntimeError("找不到 CARD_REWARD 真机帧模板，请确认 logs/realtime/ 存在")
    os.makedirs(os.path.dirname(os.path.join(_ROOT, TPL_PATH)), exist_ok=True)
    json.dump(tpl, open(os.path.join(_ROOT, TPL_PATH), "w", encoding="utf-8"),
              ensure_ascii=False)
    return tpl


# ---------------------------------------------------------------- 卡名 / 元数据

def norm_key(name: str) -> str:
    """游戏显示名 → card_data 的键格式：'Dual Wield' → 'dual_wield'。"""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace("'", "")


def card_meta(name: str) -> dict:
    """卡牌的 cost / type / rarity（供构造 screen_state.cards 用）。

    来源：models/card_data.py 的 CARD_EFFECTS（base_cost / is_attack / is_skill
    / is_power / rarity_score）。查不到时给安全默认值并计入覆盖率统计。
    """
    from models.card_data import CARD_EFFECTS
    d = CARD_EFFECTS.get(norm_key(name))
    if d is None:
        return {"cost": 1, "type": "SKILL", "rarity": "COMMON", "_hit": False}
    typ = "ATTACK" if d.get("is_attack") else ("POWER" if d.get("is_power") else "SKILL")
    rs = d.get("rarity_score", 0.5)
    rarity = "RARE" if rs >= 1.0 else ("UNCOMMON" if rs >= 0.5 else "COMMON")
    cost = d.get("base_cost", 1)
    return {"cost": int(cost) if isinstance(cost, (int, float)) else 1,
            "type": typ, "rarity": rarity, "_hit": True}


def make_card_entry(name: str, proto: dict) -> dict:
    """构造与真机帧同构的 card 字典（复制模板元素保证字段齐全）。"""
    m = card_meta(name)
    e = copy.deepcopy(proto)
    e["id"] = name
    e["name"] = name
    e["cost"] = m["cost"]
    e["type"] = m["type"]
    e["rarity"] = m["rarity"]
    e["upgrades"] = 0
    e["is_playable"] = False
    e["has_target"] = m["type"] == "ATTACK"
    return e


# ---------------------------------------------------------------- 状态重建

def deck_at(run: dict, floor: int) -> list:
    """反推"第 floor 层决策点当时"的牌组（元素如 'Strike_R' / 'Bloodletting+1'）。

    以 master_deck（最终牌组）为锚，撤销 floor 之后发生的变化：
      · 减去 floor 之后通过 card_choices / items_purchased 获得的卡
      · 加回 floor 之后被 items_purged 移除的卡
      · 撤销 floor 之后的 campfire 升级（去掉 '+N' 后缀）
    """
    deck = list(run.get("master_deck") or [])

    def _strip(c):
        return c.split("+")[0]

    # 1) 之后获得的卡（选牌奖励）
    later_gain = [c["picked"] for c in run.get("card_choices", [])
                  if c.get("floor", 0) > floor and c.get("picked")]
    # 商店买的可能是卡也可能是遗物/药水：用 card_meta 能否命中 + 命名风格判断
    purch_floors = run.get("item_purchase_floors", []) or []
    for i, item in enumerate(run.get("items_purchased") or []):
        f = purch_floors[i] if i < len(purch_floors) else 0
        if f > floor and card_meta(item)["_hit"] and " " in str(item):
            later_gain.append(item)

    # 2) 之后移除的卡
    purge_floors = run.get("items_purged_floors", []) or []
    later_purge = []
    for i, c in enumerate(run.get("items_purged") or []):
        f = purge_floors[i] if i < len(purge_floors) else 0
        if f > floor:
            later_purge.append(c)

    # 3) 之后的升级（campfire SMITH / 事件的 cards_upgraded）
    later_upgrade = []
    for cf in run.get("campfire_choices") or []:
        if cf.get("key") == "SMITH" and cf.get("floor", 0) > floor and cf.get("data"):
            later_upgrade.append(cf["data"])
    for ev in run.get("event_choices") or []:
        if ev.get("floor", 0) > floor:
            later_upgrade.extend(ev.get("cards_upgraded") or [])

    # --- 施加撤销 ---
    def _remove_one(lst, target):
        tgt = _strip(target)
        for i, c in enumerate(lst):
            if _strip(c) == tgt:
                return lst.pop(i)
        return None

    for g in later_gain:
        _remove_one(deck, g)
    for p in later_purge:
        deck.append(p)          # 当时它还在牌组里（升级状态未知，按未升级处理）
    for u in later_upgrade:
        tgt = _strip(u)
        for i, c in enumerate(deck):
            if _strip(c) == tgt and "+" in c:
                deck[i] = tgt
                break
    return deck


def deck_is_sane(deck: list) -> bool:
    """牌组合理性：张数在合理区间，且至少含基础起手牌的痕迹。"""
    return 10 <= len(deck) <= 40


# ---------------------------------------------------------------- 样本构造

class SampleBuilder:
    """把「重建的状态」变成 (obs, 候选向量, 标签)。"""

    def __init__(self):
        import slaythespire as sts
        from models.obs_encoder import enhance_observation, encode_deck_stats, OBS_DIM
        from models.card_index import CARD_INDEX
        from models.card_data import CARD_ARCHETYPE, CARD_EFFECTS
        from models.candidate_encoder import encode_candidate, DESC_VEC_DIM
        from env.base_env import Candidate

        self.sts = sts
        self._enh = enhance_observation
        self._deck_stats = encode_deck_stats
        self.OBS_DIM = OBS_DIM
        self.DESC_VEC_DIM = DESC_VEC_DIM
        self._encode = encode_candidate
        self._Candidate = Candidate
        self.ci, self.at, self.ce = CARD_INDEX, CARD_ARCHETYPE, CARD_EFFECTS
        self.tpl = get_template()
        self.nni = sts.getNNInterface()
        self.meta_hit = self.meta_miss = 0
        self.n_degenerate = 0   # 候选向量退化（互相同）的样本数

    def build(self, run: dict, cc: dict):
        """cc = 一条 card_choices 记录。返回 (obs, vecs, label, info) 或 None。"""
        floor = cc["floor"]
        picked, not_picked = cc.get("picked"), list(cc.get("not_picked") or [])
        if not picked or len(not_picked) + 1 < 2:
            return None

        deck = deck_at(run, floor)
        if not deck_is_sane(deck):
            return None

        gs = copy.deepcopy(self.tpl)
        proto = gs["screen_state"]["cards"][0]

        # 候选：人类面对的选项（picked + not_picked），**必须来自 .run 本身**
        names = [picked] + not_picked
        if len(names) > MAX_CANDS:
            # 少数局出现超长 not_picked（数据异常/特殊奖励屏）——丢弃，
            # 否则候选矩阵装不下、label 会越界（实测会触发 CUDA device assert）
            return None
        entries = []
        for n in names:
            m = card_meta(n)
            if m["_hit"]:
                self.meta_hit += 1
            else:
                self.meta_miss += 1
            entries.append(make_card_entry(n, proto))

        # 打乱顺序（Scorer 是逐候选独立打分，顺序无关；打乱可避免"第一位总是答案"偏置）
        order = list(range(len(entries)))
        random.shuffle(order)
        entries = [entries[i] for i in order]
        label = order.index(0)      # picked 原在索引 0

        hps = run.get("current_hp_per_floor") or []
        mhps = run.get("max_hp_per_floor") or []
        golds = run.get("gold_per_floor") or []
        idx = max(0, min(floor - 1, len(hps) - 1)) if hps else None

        gs["screen_type"] = "CARD_REWARD"
        gs["floor"] = floor
        gs["act"] = 1 if floor <= 16 else (2 if floor <= 33 else 3)
        gs["ascension_level"] = 0            # 目标场景是 A0（数据是 A20，见报告说明）
        gs["current_hp"] = hps[idx] if idx is not None else gs.get("max_hp", 80)
        gs["max_hp"] = mhps[idx] if idx is not None and idx < len(mhps) else gs.get("max_hp", 80)
        gs["gold"] = golds[idx] if idx is not None and idx < len(golds) else 100
        gs["choice_list"] = [e["id"] for e in entries]
        gs["screen_state"] = dict(gs["screen_state"])
        gs["screen_state"]["cards"] = entries
        gs["screen_state"]["skip_available"] = True
        gs["deck"] = [make_card_entry(c.split("+")[0], proto) | {
            "upgrades": 1 if "+" in c else 0} for c in deck]
        gs["relics"] = self._relics(run, floor)
        gs["potions"] = gs.get("potions") or []

        gc = self.sts.GameContext(self.sts.CharacterClass.IRONCLAD, 0, 0)
        gc.init_from_json(json.dumps(gs))
        raw = list(self.nni.getObservation(gc))
        obs = self._enh(raw, gc, card_index=self.ci, arch_table=self.at,
                        card_effects=self.ce)
        if len(obs) != self.OBS_DIM:
            return None
        da = self._deck_stats(raw, card_index=self.ci, arch_table=self.at,
                              card_effects=self.ce)[:5]

        # 候选向量：走**与真机完全相同的构造路径**（models/live_candidates），
        # 否则 card=None 会让候选向量丢失卡牌信息、所有候选长得一样（实测
        # loss 不下降、准确率等于随机——这是个真实踩过的坑）。
        from models.live_candidates import build_card_reward_candidates
        specs = [sp for sp in build_card_reward_candidates(gs)
                 if sp.get("kind") != "SKIP"]      # BC 不含 SKIP（见上方说明）
        if len(specs) != len(entries):
            return None                            # 构造器与我们的候选数不一致 → 丢弃
        vecs = []
        for i, sp in enumerate(specs):
            proto_c = self._Candidate(idx=i, kind=sp["kind"], desc_vec=[],
                                      label=sp["label"])
            v = self._encode(proto_c, card=sp.get("card"), room=sp.get("room"),
                             bits=sp.get("bits"), deck_arch=da)
            vecs.append(v)
        # sanity：同一样本内候选向量必须互不相同，否则网络无法区分
        if len({tuple(round(x, 6) for x in v) for v in vecs}) < len(vecs):
            self.n_degenerate += 1
            return None
        # ⚠️ BC 数据**刻意不含 SKIP 候选**：
        # `.run` 从不记录"人类跳过了哪次奖励"（`card_choices` 只记录选了牌的），
        # 若把 SKIP 放进候选，它会永远不是正确答案 → 反而教出"永不跳牌"，
        # 恰好加剧我们要修的毛病。SKIP 的偏好留给 RL 微调（V10 表征已给出可学信号）。

        # padding
        pad = [0.0] * self.DESC_VEC_DIM
        while len(vecs) < MAX_CANDS:
            vecs.append(list(pad))
        vecs = vecs[:MAX_CANDS]
        mask = [1] * len(entries)            # 真实候选数（仅卡牌）
        mask += [0] * (MAX_CANDS - len(mask))
        return (np.asarray(obs, dtype=np.float32),
                np.asarray(vecs, dtype=np.float32),
                int(label), np.asarray(mask[:MAX_CANDS], dtype=np.float32),
                {"file": run.get("_file"), "floor": floor, "picked": picked,
                 "n_cands": len(entries), "victory": bool(run.get("victory"))})

    def _relics(self, run: dict, floor: int) -> list:
        """重建 floor 之前获得的遗物（来自 .run 的 relics_obtained）。"""
        out = []
        for r in run.get("relics_obtained") or run.get("relics_obtained_log") or []:
            if isinstance(r, dict) and r.get("floor", 0) <= floor and r.get("key"):
                out.append(r["key"])
        # 起手遗物：铁甲固定 Burning Blood
        if "Burning Blood" not in out:
            out.insert(0, "Burning Blood")
        proto = None
        for r in self.tpl.get("relics") or []:
            proto = copy.deepcopy(r)
            break
        if proto is None:
            return []
        res = []
        for name in out:
            e = copy.deepcopy(proto)
            e["id"] = name
            e["name"] = name
            e["counter"] = -1
            e["used_up"] = False
            res.append(e)
        return res


# ---------------------------------------------------------------- 主流程

def load_runs(character="IRONCLAD", only_wins=False, limit=None):
    """按 `character_chosen` 字段筛选（**不能按目录名**——部分铁甲局落在
    非 IRONCLAD 命名的目录里，按目录会漏掉约一半）。"""
    pat = os.path.join(_ROOT, "data/external/Slay-the-Spire-data/runs/*/*/*.run")
    files = sorted(glob.glob(pat)) or sorted(glob.glob(
        os.path.join(_ROOT, "data/external/Slay-the-Spire-data/runs/*/*/*/*.run")))
    out = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if str(d.get("character_chosen", "")).upper() != character.upper():
            continue
        if only_wins and not d.get("victory"):
            continue
        d["_file"] = os.path.relpath(f, _ROOT)
        out.append(d)
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/human_bc/card_reward.npz")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--only-wins", action="store_true", help="只用胜局（质量更高）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    runs = load_runs(only_wins=args.only_wins, limit=args.max_runs)
    print(f"[data] 载入 {len(runs)} 局 Ironclad"
          f"{'（仅胜局）' if args.only_wins else ''}")

    b = SampleBuilder()
    OBS, VEC, LAB, MASK, INFOS = [], [], [], [], []
    n_cc = n_fail = 0
    for run in runs:
        for cc in run.get("card_choices") or []:
            n_cc += 1
            try:
                r = b.build(run, cc)
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  [warn] 构造失败 {run.get('_file')} floor={cc.get('floor')}: "
                          f"{type(e).__name__}: {str(e)[:110]}")
                continue
            if r is None:
                n_fail += 1
                continue
            obs, vecs, lab, mask, info = r
            OBS.append(obs); VEC.append(vecs); LAB.append(lab)
            MASK.append(mask); INFOS.append(info)

    print(f"[build] 决策点 {n_cc} 个 → 成功样本 {len(OBS)} 条（失败 {n_fail}，其中候选向量退化 {b.n_degenerate}）")
    if b.meta_hit + b.meta_miss:
        tot = b.meta_hit + b.meta_miss
        print(f"[meta] 卡牌元数据命中率 {b.meta_hit}/{tot} "
              f"= {b.meta_hit / tot * 100:.1f}%")
    if not OBS:
        print("!! 没有产出任何样本")
        return 1

    OBS = np.stack(OBS); VEC = np.stack(VEC)
    LAB = np.asarray(LAB, dtype=np.int64); MASK = np.stack(MASK)
    print(f"[shape] obs {OBS.shape} / cands {VEC.shape} / labels {LAB.shape}")
    print(f"[label] 分布（打乱后应接近均匀）：{np.bincount(LAB, minlength=MAX_CANDS)}")

    if args.dry_run:
        print("[dry-run] 不落盘")
        return 0

    out = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, obs=OBS, cands=VEC, labels=LAB, mask=MASK,
                        meta=np.array([json.dumps(i) for i in INFOS]))
    print(f"[save] {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
