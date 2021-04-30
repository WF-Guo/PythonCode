n = int(input())
list = [ ]
for s in input().split():
    list.append(int(s))

print(list)
x =int(input())

i = 0
while i< n:
    if int(list[i]) == x:
        print(i)
        break
    i +=1