import sys

array = []

for line in sys.stdin:
    print(line)
    if line[0] is '\n':
        break
    array.append([int(x) for x in line.split(' ')])
print(array)