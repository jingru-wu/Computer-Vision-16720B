#coding=utf-8
import sys

def search_conncted(A,index,count,neb):
    count=0
    x=index[0]
    y=index[1]
    B=[]
    for n in neb:
        if x+n[0]<0 or y+n[1]<0:
            break
        else:
            x_index=x+n[0]
            y_index=y+n[1]
        if A[x_index][y_index]==1:
            A[x_index][y_index]=0
            B.insert(0,[x_index,y_index])
            search_conncted(A,index=B.pop(),count=count,neb=neb)
            count=count+1
    return count



A=[[0,0,0,0,0,1,1,0,0,0,1],
[1,1,0,0,0,1,1,0,0,0,0],
[1,1,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,1,1,1,1,0,0],
[0,0,0,0,0,0,1,1,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0]]

H=len(A)
W=len(A[0])

neb=[[-1,-1],
     [-1,0],
     [-1,1],
     [0,-1],
     [0,1],
     [1,-1],
     [1,0],
     [1,1]]

for i in range(1,H-1):
    for j in range(1,W-1):
        if A[i][j]==1:
            search_conncted(A,[i,j],count=-1,neb=neb)






