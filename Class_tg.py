from aiogram.fsm.state import StatesGroup
from aiogram.fsm.state import State, StatesGroup

class StocksState(StatesGroup):
    stock_situation = State()
    stock_card = State()
    stock_combination = State()


# class TaroBot:
#     def __init__(self):
#         self.text = None
#
#     def answer(self, text):
#         self.text = text[::-1]
#         return self.text