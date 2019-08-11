# -*- coding:utf-8 -*-


def karatsuba(num1, num2):
    # 求n
    digits1 = len(str(int(num1)))
    digits2 = len(str(int(num2)))
    n = min(digits1, digits2) / 2

    if n == 0:
        return num1 * num2

    # 求 a, b, c, d
    a = num1 / (10 ** n)
    b = num1 % (10 ** n)
    c = num2 / (10 ** n)
    d = num2 % (10 ** n)

    # step1 -> ac
    step1 = karatsuba(a, c)

    # step1 -> bd
    step2 = karatsuba(b, d)

    # step3 -> (a + b)(c + d) - step1 - step2
    step3 = karatsuba(a + b, c + d) - step1 - step2

    return step1 * (10 ** (2 * n)) + step3 * (10 ** n) + step2


if __name__ == '__main__':
    num1 = 1234
    num2 = 5678
    print karatsuba(num1, num2)
