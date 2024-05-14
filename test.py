class A:
    def __init__(self) -> None:
        self.a = 1
    def func(self):
        self.b = 2

c = A()
print(c.b)
c.func()
print(c.b)