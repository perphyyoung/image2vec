def foo(i):
    return i


def foo2(i):
    return i * 2


def foo3(i):
    return i * 3


lst_func = ['foo', 'foo2', 'foo3']
for ind in range(3):
    print(eval(lst_func[ind])(14))
