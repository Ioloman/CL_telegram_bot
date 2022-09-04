import asyncio
import os
from aiogram import Bot, Dispatcher, types
from dotenv import load_dotenv
from question import question_handler
import nltk


async def start_handler(event: types.Message):
    """
    Handler for /start command
    """
    await event.answer(
        f"Привет, {event.from_user.get_mention(as_html=True)} 👋😎!\n❓Задавай вопрос❓",
        parse_mode=types.ParseMode.HTML,
    )


async def main():
    bot = Bot(token=os.getenv('BOT_TOKEN'))
    nltk.download('punkt')

    try:
        disp = Dispatcher(bot=bot)
        disp.register_message_handler(start_handler, commands={"start"})
        disp.register_message_handler(question_handler)
        await disp.start_polling()
    finally:
        await bot.close()


if __name__ == '__main__':
    load_dotenv('.env')
    asyncio.run(main())



