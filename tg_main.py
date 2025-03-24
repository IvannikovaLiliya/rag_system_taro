import asyncio
import logging
from aiogram import Bot, Dispatcher
from rag_system_taro.app.handlers import router


from rag_system_taro.taro import RagModel

ragm = RagModel()
# ragm.rag_luanch(query)




async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token='7870948069:AAGM7p4-yz8Nm93vZvvIYnFfN937wU0IwB4')
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot, skip_updates=True)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Бот выключен')