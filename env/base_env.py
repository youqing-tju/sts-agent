"""环境抽象：所有环境实现（模拟器/真实游戏）都遵守这个接口。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class Obs:
    vec: list          # 412 维 NN 观测（lightspeed NNInterface 输出）
    hp: int
    max_hp: int
    floor: int
    gold: int
    deck: list         # 卡牌 id 列表（可读形式，给规则策略和日志用）
    relics: list
    screen: str        # MAP / CARD_REWARD / SHOP / REST / EVENT / BOSS_REWARD ...

@dataclass
class Candidate:
    idx: int           # 动作索引（传给 step）
    kind: str          # MAP_NODE / CARD_REWARD / SKIP / SHOP_ITEM / REST_OPTION / EVENT_OPTION
    desc_vec: list     # 候选描述向量（M2 给打分网络用）
    label: str         # 可读名称，日志用

@dataclass
class StepResult:
    obs: Obs
    done: bool
    info: dict = field(default_factory=dict)   # 战斗 HP 进出、战斗类型等

class StsEnv(ABC):
    @abstractmethod
    def reset(self, seed: int) -> Obs: ...
    @abstractmethod
    def observe(self) -> Obs: ...
    @abstractmethod
    def candidates(self) -> list: ...          # 当前可选动作（空列表 = 战斗中，MCTS 接管）
    @abstractmethod
    def step(self, idx: int) -> StepResult: ...
