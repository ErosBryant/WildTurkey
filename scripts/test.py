import sys
import telegram
import asyncio
import datetime

bot = telegram.Bot(token='7098217591:AAE2y9o-obMbKP2fVBJGlk8luXt3ImHukW8')
chat_id = 6948187826
nowtime = datetime.datetime.now()


arg1 = sys.argv[1] #Start or End
arg2 = sys.argv[2] #Work name
msg = arg1 + ", " + arg2 + "\nTime : " + nowtime.strftime("%Y-%m-%d %H:%M:%S")

asyncio.run(bot.sendMessage(chat_id=chat_id, text=msg))