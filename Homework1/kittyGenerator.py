def kittyGenerator():
    i = 1
    while True:
        yield i
        i = i + i
for x in kittyGenerator():
    if x > 12:
        break
    meowList = ['Meow' for y in range(0, x)]
    print(meowList)