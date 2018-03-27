def assertRaises(arg):
    if issubclass(arg, Exception):
        def inner(fn):
            try:
                fn()
            except arg:
                print(True)
            else: print(False)
        return inner
    else:
        try:
            fn()
        except:
            print(True)
        else:
            print(False)

@assertRaises(TypeError)
def f():
    del str.partition

@assertRaises(TypeError)
def f():
    property.start = 0

@assertRaises(TypeError)
def f():
    def inner(a, b):
        print(a, b)
    inner(1, **{len: 2})
