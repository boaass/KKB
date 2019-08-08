# coding=utf-8

# You are playing a card game with your friends. This game in China named “扎金花”. In this game, the 2, 3, 5 are some
# simple powerful numbers. Because the combination of 2,3,5 is less than any other combinations but greater than the
# AAA, which is the king in this game. In today, you want to find if a number is a simple number, in which their
# factors only include 2, 3 and 5. So your task is to find out whether a given number is an amazing number. E.g
# Input: 6 Output: (2, 3) Explanation: 6 = 2 x 3

# Input: 8 Output: (2, 2, 2) Explanation: 8 = 2 x 2 x 2
# Input: 14 Output:None Explanation: 14 is not amazing since it includes another prime factor 7.

# How to check your answer:
# If you test 1845281250, your program should give (2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5);
# If you test 3690562500, your program should give (2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5);
# If you test 1230187500, your program should give (2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5);
# If you test 10023750, your program should give None;


def findAmazingNumber(num):
    amazingNums = [2, 3, 5]
    r = []
    while True:
        temp_num = num
        for n in amazingNums:
            if temp_num % n == 0:
                num /= n
                r.append(n)
        if num == temp_num:
            return None
        elif num == 1:
            return tuple(sorted(r))


if __name__ == '__main__':
    print findAmazingNumber(6)
    print findAmazingNumber(8)
    print findAmazingNumber(14)
    print findAmazingNumber(1845281250)
    print findAmazingNumber(3690562500)
    print findAmazingNumber(1230187500)
    print findAmazingNumber(10023750)
