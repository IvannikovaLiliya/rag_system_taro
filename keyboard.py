from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

knopki = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='🔮Расклад'),KeyboardButton(text='🃏Значение карты'),KeyboardButton(text='♠️Значение комбинации')],
                                       [KeyboardButton(text='📊Статистика')]],
                           resize_keyboard=True, input_field_placeholder='Выберите кнопку')

like_dislike = InlineKeyboardMarkup(inline_keyboard=
            [[InlineKeyboardButton(text='👍', callback_data='like'),
            InlineKeyboardButton(text='👎', callback_data='dislike')],
             [InlineKeyboardButton(text='Продолжить', callback_data='Continue')]])

cont_comp = InlineKeyboardMarkup(inline_keyboard=
            [[InlineKeyboardButton(text='Продолжить', callback_data='Continue'),
            InlineKeyboardButton(text='Завершить', callback_data='Complete')]])

taro = InlineKeyboardMarkup(inline_keyboard=
            [[InlineKeyboardButton(text='🔮Расклад', callback_data='Layout'),
            InlineKeyboardButton(text='🃏Значение карты', callback_data='Card meaning'),
            InlineKeyboardButton(text='️♠️Значение комбинации', callback_data='Combination value')]])