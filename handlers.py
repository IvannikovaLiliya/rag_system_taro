import logging
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
import rag_system_taro.app.keyboard as kb
from aiogram.fsm.state import State, StatesGroup
# from .Class_tg import TaroBot
import os
import sys
sys.path.append(os.path.abspath("rag_system_taro"))

from rag_system_taro.taro import RagModel


ragm = RagModel()





# Подключаем маршрутизатор
router = Router()

# Моделька
ragm = RagModel()

ratings_count = {"like": 0,
                 "dislike": 0,
                 }



class StocksState(StatesGroup):
    situation = State()
    card = State()
    combination = State()



@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer('Привет я бот, который умеет: Делать расклад по твоей ситуации.'
                         ' Так же могу ответить, что значит каждая карта, название который ты мне отправишь (Колода Уайта) и расскажу,'
                         ' что значит сочитания 2х карт из этой же колоды',reply_markup=kb.knopki)




@router.message(F.text == '🔮Расклад')
async def rasklad(message: Message, state: FSMContext):
    await state.set_state(StocksState.situation)
    await message.answer('Опиши ситуацию.')

@router.message(StocksState.situation)
async def mean_comb(message: Message, state: FSMContext):
    await state.update_data(situation=message.text)
    text = await state.get_data()
    flag = 1
    taro = RagModel()
    # await message.answer(f'Ваша ситуация: {text["situation"]}')
    answer = taro.rag_luanch(text["situation"], flag)
    await message.answer(f'Расклад по вашей ситуации: {answer}',reply_markup=kb.like_dislike)
    await state.clear()




@router.message(F.text == '🃏Значение карты')
async def card(message: Message, state: FSMContext):
    await state.set_state(StocksState.card)
    await message.answer('Опиши карту.')

@router.message(StocksState.card)
async def mean_comb(message: Message, state: FSMContext):
    await state.update_data(card=message.text)
    text = await state.get_data()
    flag = 0
    taro = TaroBot()
    # await message.answer(f'Ваша карта: {text["card"]}')
    answer = taro.rag_luanch(text['card'],flag)
    await message.answer(f'Вот значение карты: {answer}',reply_markup=kb.like_dislike)
    await state.clear()




@router.message(F.text == '♠️Значение комбинации')
async def combination(message: Message, state: FSMContext):
    await state.set_state(StocksState.combination)
    await message.answer('Опиши комбинацию.')



@router.message(StocksState.combination)
async def mean_comb(message: Message, state: FSMContext):
    await state.update_data(combination=message.text)
    text = await state.get_data()
    flag = 0
    taro = TaroBot()
    # await message.answer(f'Ваша комбинация: {text["combination"]}')
    answer = taro.rag_luanch(text["combination"],flag)
    await message.answer(f'Вот значение комбинации: {answer}',reply_markup=kb.like_dislike)
    await state.clear()


# @router.callback_query(F.data.in_(['like','dislike','Continue']))
# async def answer_metric(callback: CallbackQuery):
#     # await callback.message.answer('Что вы хотите дальше',reply_markup=kb.cont_comp)
#     await callback.message.edit_reply_markup(reply_markup=None)
#     await callback.message.answer("Что вы хотите дальше?", reply_markup=kb.knopki)


@router.callback_query(F.data.in_(["like", "dislike"]))
async def answer_metric(callback: CallbackQuery):
    global ratings_count
    rating = callback.data

    ratings_count[rating] += 1

    await callback.answer("Спасибо за вашу оценку!")

    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Что вы хотите дальше?", reply_markup=kb.knopki)


@router.callback_query(F.data.in_(["Continue"]))
async def answer_cont(callback: CallbackQuery):
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Что вы хотите дальше?", reply_markup=kb.knopki)




@router.message(F.text == "📊Статистика")
async def stats_command(message: Message):
    likes = ratings_count["like"]
    dislikes = ratings_count["dislike"]
    await message.answer(f"👍 Лайков: {likes}\n👎 Дизлайков: {dislikes}\n")






@router.message()
async def unknown_message(message: Message):
    await message.answer("🤔 Я вас не понимаю.Выберите кнопку.", reply_markup=kb.knopki)




async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token='7870948069:AAGM7p4-yz8Nm93vZvvIYnFfN937wU0IwB4')
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot, skip_updates=True)
