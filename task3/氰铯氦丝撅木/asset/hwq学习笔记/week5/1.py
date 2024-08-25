import numpy as np 

Δt1 = 2 - 0
Δt2 = 4 - 2
Δt3 = 9 - 4

T = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, Δt1, Δt1**2, Δt1**3, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, Δt2, Δt2**2, Δt2**3, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, Δt3, Δt3**2, Δt3**3],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2*Δt3, 3*Δt3**2],
     [0, 1, 2*Δt1, 3*Δt1**2, 0, -1, 0, 0, 0, 0, 0, 0],
     [0, 0, 2, 6*Δt1, 0, 0, -2, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 2*Δt2, 3*Δt2**2, 0, -1, 0, 0],
     [0, 0, 0, 0, 0, 0, 2, 6*Δt2, 0, 0, -2, 0] ]  ## 需要仔细 检查， 很容易 打错

def getA(θ):
    θ = np.array(θ)
    A = np.dot(np.linalg.inv(T), θ.T)
    A = np.around(A, decimals = 2)  ## 结果 保留 到 小数点 后 两位
    return A 

## X 的导数 为 速度， 初始和末尾的速度均为0
X = [-4, -5, -5, 2, 2, 2, 0, 0, 0, 0, 0, 0]
print('X_A：')
print(getA(X))

## Y 的导数 为 速度， 初始和末尾的速度均为0
Y = [0, 5, 5, 3, 3, -3, 0, 0, 0, 0, 0, 0]
print('\nY_A：')
print(getA(Y))

## θ 
θ = [120, 45, 45, 30, 30, 0, 0, 0, 0, 0, 0, 0]
print('\nθ_A：')
print(getA(θ))
