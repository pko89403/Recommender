from random import * 
# Python 3
"""
    Freivald's algorithm
    확률적 랜덤화 알고리즘 매트릭스 곱을 검증하기 위한
    A, B, C 가 모두 n x n 정방 행렬일 때. A X B = C를 일반적으로 검증하려고 할때 
    결국 시간 복잡도 감소하려고. 높은 확률로 
    A X (Br) - (Cr) = 0
"""
"""
    R에서 nxn의 부분 정방행렬을 뽑는 방법
"""
def r_calc(input, r, n):
    inputr = [0] * n

    for i in range(n):
        for j in range(n):
            inputr[i] += ( input[i][j] * r[j])
    
    return inputr

def freivald_calc(a,b,c,n):
    # generate random (n, 0) shape matrix ... filled with 0 or 1
    
    r = [randint(0,1) for i in range(n)]
    while(sum(r) == 0):
        r = [randint(0,1) for i in range(n)]
    
    br = r_calc(b, r, n)
    cr = r_calc(c, r, n)
    axbr = r_calc(a, br, n)

    for i in range(0, n):
        if(axbr[i] != cr[i]): # False
            return False

    return True

def get_submatrix(input, n, m, k):
    submatrix_list = []
    for i in range(0, n-k+1):
        for j in range(0, m-k+1):
            #print(i, j)
            temp = []
            for e in range(k):
                temp.append(input[i+e][j:j+k])
            
            submatrix_list.append(temp)            
    return submatrix_list



T = int(input())
for t in range(T):
    temp_input = input().split(' ')
    N = int(temp_input[0])
    M = int(temp_input[1])
    K = int(temp_input[2])

    X = []
    Y = []
    R = []

    for k in range(K):
        temp = input().split(' ')
        temp = [int(v) for v in temp]
        X.append(temp)
    for k in range(K):
        temp = input().split(' ')
        temp = [int(v) for v in temp]
        Y.append(temp)
    for n in range(N):
        temp = input().split(' ')
        temp = [int(v) for v in temp]
        R.append(temp)
    
    submatrix = get_submatrix(R, N, M, K)
    submatrix= sorted(submatrix)

    #print(submatrix)
    answer = []
    for idx, b in enumerate(submatrix):
        check=True

        if(idx > 0):
            if(b == submatrix[idx-1]):
                answer.append(answer[-1])
                continue 

        for trials in range(3):
            check = freivald_calc(X,b,Y,K)
            if(check == False):
                break
        #print(b, check)
        answer.append(check)
    print(sum(answer))