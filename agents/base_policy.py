"""Policy 抽象接口：宏观决策策略（选路/卡牌奖励/商店/篝火/事件）。"""
from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def choose(self, obs, candidates: list) -> int:
        """从 candidates 中选一个，返回其 idx。"""
        ...
