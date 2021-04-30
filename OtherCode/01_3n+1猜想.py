if __name__ == '__main__':

    count = 0
    n = int(input())
    while n != 1:
        if n %2 == 0:
            n /=2
        else:
            n = (3*n+1)/2
        count+=1
    print(count)