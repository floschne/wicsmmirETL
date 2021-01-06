from abc import ABC, abstractmethod
from typing import Union

from pandas import DataFrame


class FilterBase(ABC):

    def __init__(self, column_id: Union[int, str]):
        self.cId = column_id

    @abstractmethod
    def filter(self, x) -> bool:
        pass

    def __call__(self, df: DataFrame) -> bool:
        return self.filter(df[self.cId])
