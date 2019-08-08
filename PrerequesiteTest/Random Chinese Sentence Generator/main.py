# -*- coding:utf-8 -*-

# Writing a programming which could generate random Chinese sentences based on one grammar. Your input grammar is:
# simple_grammar = """
# sentence => noun_phrase verb_phrase noun_phrase => Article Adj* noun
# Adj* => null | Adj Adj*
# verb_phrase => verb noun_phrase
# Article => 一个 | 这个
# noun => 女人| 篮球|桌子|小猫
# verb => 看着 | 听着 | 看见
# Adj=> 蓝色的| 好看的|小小的|年轻的 """

# Your task is define a function called generate, if we call generate(‘sentence’), you could see some sentences like:
# >> generate(“sentence”)
# Output: 这个蓝色的女人看着一个小猫
# >> generate(“sentence”)
# Output: 这个好看的小猫坐在一个女人

import random


Article = ["一个", "这个"]
noun = ["女人", "篮球", "桌子", "小猫"]
verb = ["看着", "听着", "看见"]
Adj = ["蓝色的", "好看的", "小小的", "年轻的"]


def generate(command):

    if command == 'sentence':
        print Article[random.randint(0, len(Article)-1)] + \
              Adj[random.randint(0, len(Adj)-1)] + \
              noun[random.randint(0, len(noun)-1)] + \
              verb[random.randint(0, len(verb)-1)] + \
              Article[random.randint(0, len(Article)-1)] + \
              noun[random.randint(0, len(noun)-1)]
    else:
        print 'command error ...'

if __name__ == '__main__':
    generate('sentence')
    generate('sentence')
    generate('sentence')
    generate('sentence')