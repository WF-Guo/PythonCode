if __name__ == '__main__':

    n = int(input())
    SUM = [0 for i in range(n)]
    for i in range(n):
        a,b = map(int,input().split())
        SUM[a] += b
    k = 0
    MAX = 0
    for i in range(n):
        if MAX<SUM[i]:
            MAX=SUM[i]
            k = i
    print(k,end=' ')
    print(MAX)


