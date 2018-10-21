import numpy as np 


# 卡尔曼滤波步骤
# 1 测量更新
def update(mean1, var1, mean2, var2):
    new_mean =(var2*mean1 + var1*mean2) / (var1+var2)
    new_var = (var1*var2) / (var1+var2)
    return [new_mean, new_var]

#print(update(0, 1, 0, 1))

# 2 运动更新/预测步骤/时间更新
def predict(mean1, var1, mean2, var2):
    return [mean1+mean2, var1+var2]

# 3 结合测量更新和运动更新，实现测量-运动的循环


measurements = [5, 6, 7, 9, 10]
measurements_sig = 4


motion = [1, 1, 2, 1, 1]
motion_sig = 2

mu = 0
sig = 10000
update_mu = []
update_sig = []
predict_mu = []
predict_sig = []

for i in range(len(measurements)):
    [mu, sig] = update(mu, sig, measurements[i], measurements_sig)
    update_mu.append(mu)
    update_sig.append(sig)
    [mu, sig] = predict(mu, sig, motion[i], motion_sig)
    predict_mu.append(mu)
    predict_sig.append(sig)
#print("update mu:", update_mu)
#print("update sigma:", update_sig)
#print("predict mu:", predict_mu)
#print("predict sigma:", predict_sig)


# 状态和面向对象编程
# 1 恒定速度模型
def predict_state(state, dt):
    x, vel = state
    x = x + vel*dt
    return [x, vel]

#print(predict_state((10, 5), 2))

# 2 恒定加速度模型

# 3 通过类跟踪汽车的状态
import matplotlib.pyplot as plt 
class Car(object):
    # 定义构造函数
    def __init__(self, position, velocity, world, color="r"):
        # 初始化状态
        # position is a list [y, x] and so is velocity [vy, vx]
        self.state = [position, velocity]
        self.world = world
        self.color = color
        # 初始化路径
        self.path = []
        # 这条语句是干什么用的？
        self.path.append(position)

    def move(self, dt=1):
        hight = len(self.world)
        width = len(self.world[0])
        position = self.state[0]
        velocity = self.state[1]
        # 预测新的位置
        predict_position = [(position[0] + velocity[0]*dt)%hight, 
        (position[1] + velocity[1]*dt)%width]
        # 更新状态
        self.state = [predict_position, velocity]
        self.path.append(predict_position)

    def turn_left(self):
        velocity = self.state[1]
        predict_velocity = [-velocity[1], velocity[0]]
        self.state[1] = predict_velocity

    def turn_right(self):
        velocity = self.state[1]
        predict_velocity = [velocity[1], -velocity[0]]
        self.state[1] = predict_velocity        
 
    def display_world(self):
        position = self.state[0]
        plt.matshow(self.world, cmap="gray")
        ax = plt.gca()
        rows = len(self.world)
        cols = len(self.world[0])
        ax.set_xticks([x-0.5 for x in range(1, cols)], minor=True)
        ax.set_yticks([y-0.5 for y in range(1, rows)], minor=True)

        plt.grid(which="minor", ls="-", lw=2, color="gray")

        ax.text(position[1], position[0], 'x', ha='center', va='center', color=self.color, fontsize=30)
        # Draw path if it exists
        if(len(self.path) > 1):
        # loop through all path indices and draw a dot (unless it's at the car's location)
            for pos in self.path:
                if(pos != position):
                    ax.text(pos[1], pos[0], '.', ha='center', va='baseline', color=self.color, fontsize=30)

        # Display final result
        plt.show()

    def __repr__(self):
        return "state is {}".format(self.state)

    def __add__(self, other):
        return(np.array(self.state) + np.array(other.state))


world = np.zeros([10, 10])
car1 = Car([0, 0], [0, 1], world)
car2 = Car([1, 1], [1, 0], world)
#print(car1.state)
#car1.move(dt=2)
#car1.turn_right()
#car1.move(dt=1)
#car1.display_world()




# 4 矩阵和状态转换

# 向量编程

# 向量加法：更新状态向量。初始状态x0=[x, y, vx, vy]，
# 变化量xdelta=[x', y', vx', vy']，
# 更新后的状态x1 = x0 + xdelta

x0 = [5, 2, 10, 0]
xdelta = [3, 5, 2, 5]

# 方法1：列表生成式
def add(vector_1, vector_2):
    x1 = [x0[i]+xdelta[i] for i in range(len(x0))]
    return x1

# 方法2：numy数组
def add(vector_1, vector_2):
    vector_1, vector_2 = np.array(vector_1), np.array(vector_2)
    # 或者vector_1, vector_2 = np.array((vector_1, vector_2))
    return vector_1+vector_2

#print(add(x0, xdelta))

# 标量乘法：转换单位。比如把米/秒转换成英尺/秒。
#meters_to_feet = 1.0 / 0.3048

# 方法1：列表生成式
def multiply(scalar, vector):  
    x1feet =[vector[i]*scalar for i in range(len(vector))]
    return x1feet

# 方法2：numpy数组
def multiply(scalar, vector):
    x1feet = np.array(vector) * scalar
    return x1feet

#print(multiply(meters_to_feet, [1, 2, 3]))


# 点积。知道目前状态x1，预测一定时间后的状态x2。
# 比如车辆速度不变，预测2秒后的状态。
# x1 = x1 + vx*t，y2 = y1 + vy*t，
# 用点积可是表示为：
# x1 = [8, 7, 12, 5].[1, 0, 2, 0]
# y1 = [8, 7, 12, 5].[0, 1, 0, 2]

vector_1 = [8, 7, 12, 5]
vector_2 = [1, 0, 2, 0]

vector_3 = [1, 2, 3]

# 方法1：列表生成式
def dot_product(vectora, vectorb):
    # 检查点乘的两个向量是否长度相同
    if len(vectora) != len(vectorb):
        print("Error! Vectors must have the same length!")
        return None
    result = sum([vectora[i]*vectorb[i] for i in range(len(vectora))])
    return result

# 方法2：numpy数组
def dot_product(vectora, vectorb):
    try:
        result = np.dot(vectora, vectorb)
        return result
    except ValueError as e:
        print("ValueError:", e)

#print(dot_product(vector_1, vector_2))



# 矩阵编程
# 矩阵加法

# 方法1：for循环
def matrix_addition1(matrixA, matrixB):
    matrixSum = []
    row = []
    for i in range(len(matrixA)):
        for j in range(len(matrixA[0])):
            sum_of_matrix = matrixA[i][j] + matrixB[i][j]
            row.append(sum_of_matrix)
        matrixSum.append(row)
        row = []
    return matrixSum 


# 方法2：列表生成式
def matrix_addition2(matrixA, matrixB):
    if not len(matrixA) == len(matrixB) or not len(matrixA[0]) == len(matrixB[0]):
        raise Exception("Matrices' shape should be same.")
    result = [[matrixA[i][j]+matrixB[i][j] for j in range(len(matrixA[0]))] for i in range(len(matrixA))]
    return result


# 方法3：numpy数组
def matrix_addition3(matrixA, matrixB):
    return np.array(a) + np.array(b)

a = [
    [2,5,1],
    [6,9,7.4],
    [2,1,1],
    [8,5,3],
    [2,1,6],
    [5,3,1]
]

b = [
    [7, 19, 5.1],
    [6.5,9.2,7.4],
    [2.8,1.5,12],
    [8,5,3],
    [2,1,6],
    [2,33,1]
]


#print(matrix_addition1(a, b))
#print(matrix_addition2(a, b))
#print(matrix_addition3(a, b))


# 标量乘法
# 方法1：列表生成式
def multiply(scalar, matrix):
    return [[matrix[i][j]*scalar for j in range(len(matrix[0]))] \
    for i in range(len(matrix))]

# 方法2：numpy数组
def multiply(scalar, matrix):
    return scalar*np.array(matrix)

a = [
    [2,5,1],
    [6,9,7.4],
    [2,1,1],
    [8,5,3],
    [2,1,6],
    [5,3,1]
]

#print(multiply(10, a))


# 矩阵乘法

# 方法1：for循环
def get_row(matrix, row_number):
    row = matrix[row_number]
    return row

def get_column(matrix, column_number):
    column = [matrix[i][column_number] for i in range(len(matrix))]
    return column

def dot_product(vectora, vectorb):
    # 检查点乘的两个向量是否长度相同
    if len(vectora) != len(vectorb):
        print("Error! Vectors must have the same length!")
        return None
    result = sum([vectora[i]*vectorb[i] for i in range(len(vectora))])
    return result

def matrix_multiplication1(matrixA, matrixB):
    m_rows = len(matrixA)
    p_columns = len(matrixB[0])
    result = []
    row_result = []
    for i in range(m_rows):
        for j in range(p_columns):
            temp = dot_product(get_row(matrixA, i), get_column(matrixB, j))
            row_result.append(temp)
        result.append(row_result)
        row_result = []
    return result  


# 方法2：列表生成式
def matrix_multiplication2(matrixA, matrixB):
    if len(matrixA) != len(matrixB[0]):
        raise Exception("Rows of matrixA should equal columns of matrixB")
    # 获得矩阵B的列
    T_of_matrixB = [[matrixB[i][j] for i in range(len(matrixB))] \
    for j in range(len(matrixB[0]))] 

    # 矩阵A的每一行，和矩阵B的转置的每一行，进行向量的内积计算
    n = len(matrixA)
    result = [[dot_product(matrixA[i], T_of_matrixB[j]) for j in range(n)] \
        for i in range(n)]
    return result


a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

b = [[1, 4],
     [2, 5],
     [3, 6]]

# 方法3：numpy数组
def matrix_multiplication3(matrixA, matrixB):
    return np.dot(np.array(a), np.array(b))


#print(matrix_multiplication1(a, b))
#print(matrix_multiplication2(a, b))
#print(matrix_multiplication3(a, b))


# 矩阵转置

# 方法1：for循环
def transpose1(matrix):
    matrix_transpose = []
    row = []
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        matrix_transpose.append(row)
        row = []
    return matrix_transpose

# 方法2：列表生成式
def transpose2(matrix):
    T_of_matrix = [[matrix[i][j] for i in range(len(matrix))] \
    for j in range(len(matrix[0]))] 
    return T_of_matrix 

# 方法3：numpy数组
def transpose3(matrix):
    return np.array(matrix).T

#print(transpose1(a))
#print(transpose2(a))
#print(transpose3(a))



# 单位矩阵

# 方法1：for循环
def identity_matrix1(n):
    identity = []
    row = []
    for i in range(n):
        for j in range(n):
            if i == j:
                aij = 1
            else:
                aij = 0
            row.append(aij)
        identity.append(row)
        row = []
    return identity


# 方法2：列表生成式
def identity_matrix2(n):
    result = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
    return result

# 方法3：numpy数组
def identity_matrix3(n):
    return np.eye(n)


#print(identity_matrix1(4))
#rint(identity_matrix2(4))
#print(identity_matrix3(4))



# 矩阵的逆。如果矩阵A存在逆矩阵，矩阵A必需是方阵；
# 对于二阶方阵，两条对角线的差不能为0。
# 这里只考虑1、2阶矩阵的逆矩阵。
# 方法1
def inverse_matrix1(matrix):
    if len(matrix) != len(matrix[0]):
        raise ValueError('The matrix must be square')
    if len(matrix) > 2:
        raise ValueError("Rows and columns can't bigger than 2.")
    if len(matrix) == 1:
        return 1 / matrix[0][0]
    if len(matrix) == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        n = a*d - b*c
        if n == 0:
            raise ZeroDivisionError("Zero.")
        else:
            inverse = [[1/n * d, -1/n * b], [-1/n * c, 1/n * a]]
        return inverse 



# 方法2：matrix数组
def inverse_matrix2(matrix):
        if len(matrix) != len(matrix[0]):
            raise ValueError('The matrix must be square')
        if len(matrix) > 2:
            raise ValueError("Rows and columns can't bigger than 2.")
        else:
            return np.matrix(matrix).I

a = [[1, 2], [3, 4]]
print(inverse_matrix1(a))
print(inverse_matrix2(a))



