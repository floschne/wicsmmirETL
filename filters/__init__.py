from filters.filter_base import FilterBase


class TokenLenFilter(FilterBase):

    def __init__(self, min_num: int = 10, max_num: int = 150):
        super().__init__('num_tok')
        self.min_num = min_num
        self.max_num = max_num

    def filter(self, x) -> bool:
        # https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
        return (x > self.min_num) & (x < self.max_num)


class MinSentenceLenFilter(FilterBase):

    def __init__(self, min_len: int = 5):
        super().__init__('min_sent_len')
        self.min_len = min_len

    def filter(self, x) -> bool:
        return x > self.min_len
