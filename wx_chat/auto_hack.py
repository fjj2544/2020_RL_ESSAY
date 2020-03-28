#
from __future__ import unicode_literals
from threading import Timer
from wxpy import *
import requests

# bot=Bot()
# 从缓存中获取登录信息，刚登陆过，无需一直登陆
bot = Bot(cache_path=True)


def get_news():
    """获取金山词霸每日一句，英文和翻译"""

    url = "http://open.iciba.com/dsapi/"
    r = requests.get(url)
    content = r.json()['content']
    note = r.json()['note']
    return content, note


def send_news():
    try:
        contents = get_news()
        # 你朋友的微信名称，不是备注，也不是微信帐号。
        my_friend = bot.friends().search("小管家")[0]
        # my_friend.send(contents[0])
        # my_friend.send(contents[1])
        my_friend.send(u"今天可以把PDF版的论文给我了么?。。")
        # 每86400秒（1天），发送1次
        t = Timer(300, send_news)
        t.start()
    except:

        # 你的微信名称，不是微信帐号。

        my_friend = bot.friends().search('烟雨江南')[0]
        my_friend.send(u"今天消息发送失败了")


if __name__ == "__main__":
    send_news()