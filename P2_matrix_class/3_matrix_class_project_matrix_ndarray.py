import numbers
import numpy as np

class Matrix(object):
    # 构造矩阵
    def __init__(self, grid):
        self.g = np.array(grid)
        self.h = len(grid)
        self.w = len(grid[0])

    # 单位矩阵
    def identity(self, n):
        return Matrix(np.eye(n))

    # 矩阵的迹
    def trace(self):
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")
        else:
            return self.g.trace()

    def inverse(self):
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")
        if self.h == 1:
            m = Matrix([[1/self[0][0]]])
            return m
        if self.h == 2:
            try:
                m = Matrix(np.matrix(self.g).I)
                return m
            except np.linalg.linalg.LinAlgError as e:
                print("Determinant shouldn't be zero.", e)

    def T(self):
        T = self.g.T
        return Matrix(T)
                

    def is_square(self):
        return self.h == self.w

    def __getitem__(self,idx):
        return self.g[idx]

    def __repr__(self):
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self,other):
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 
        else:
            return Matrix(self.g + other.g)

    def __neg__(self):
        return Matrix(-self.g)

    def __sub__(self, other):
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be subtracted if the dimensions are the same") 
        else:
            return Matrix(self.g - other.g)

    def __mul__(self, other):
        if self.w != other.h:
            raise(ValueError, "number of columns of the pre-matrix must equal the number of rows of the post-matrix")    
        return Matrix(np.dot(self.g, other.g))
        
                            
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Matrix(other * self.g)

