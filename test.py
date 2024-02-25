
class Point:
    def __init__(self, coords, dim, cluster):
        self.coords = coords
        self.dim = dim
        self.cluster = cluster


def main():
    p = Point([1,2],2,-1)
    print(type(p))
    print(isinstance(p,Point))

main()