"""内置容器接入清单；格式、语义和状态均由各提供方声明。"""

from .goals import CONTAINER_CONTRACT as GOAL_CONTRACT
from .todo import CONTAINER_CONTRACT as TODO_CONTRACT


CONTRACTS = (GOAL_CONTRACT, TODO_CONTRACT)
