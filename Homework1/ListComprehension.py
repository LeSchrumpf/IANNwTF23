squareList = [x**2 for x in range(0, 100)]
evenSquareList = [x for x in squareList if x % 2 == 0]
print("Any:", squareList)
print("Only even:", evenSquareList)