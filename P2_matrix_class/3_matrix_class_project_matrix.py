import math
from math import sqrt
import numbers

def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)

def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I

class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################
 
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError, "Calculating determinant not implemented for matrices largerer than 2x2.")
        
        # TODO - your code here
        if self.h == 1:
            return self[0][0]
        if self.h == 2:
            return (self[0][0]*self[1][1]) - (self[0][1]*self[1][0])

    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        # TODO - your code here
        else:
            return sum([self[i][i] for i in range(self.h) for j in range(self.h) if i==j])

    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")

        # TODO - your code here
        if self.h == 1:
            m = Matrix([[1/self[0][0]]])
            return m
        if self.h == 2:
            if self.determinant() == 0:
                raise ZeroDivisionError("Matrix does not have an inverse if determinant equal to zero")
            else:
                a = self[0][0]
                b = self[0][1]
                c = self[1][0]
                d = self[1][1]
                temp = 1/self.determinant()
                inverse = [[temp*d, -temp*b],
                           [-temp*c, temp*a]
                          ]
                return Matrix(inverse)
        # 审阅者的代码，结果是错的
        #if self.h == 1:
            #return 1/self.determinant()
        #if self.h == 2:
            #temp = [[((-1)*(i+j))*self[1-i][1-j]/self.determinant() for j in range(self.w)] for i in range(self.h)]
            return Matrix(temp).T()




    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        # TODO - your code here
        T = [[self[i][j] for i in range(self.h)] for j in range(self.w)]
        return Matrix(T)
                

    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self,other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 
        #   
        # TODO - your code here
        #
        #matrix_add = [[self.g[i][j]+other.g[i][j] for j in range(self.w)] for i in range(self.h)]
        #m = matrix.Matrix(matrix_add)
        #return m

        # 审阅者建议
        temp = [[(self[i][j]+other[i][j]) for j in range(self.w)] for i in range(self.h)]
        return Matrix(temp) 
        # 我的想法：可以借用前面定义的重载运算符 __getitem__(self,idx)



        
        
    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        #   
        # TODO - your code here
        #
        matrix_neg = [[-1*self[i][j] for j in range(self.w)] for i in range(self.h)]

        return Matrix(matrix_neg)

        


        #return self

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        #   
        # TODO - your code here
        #
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same")
        matrix_sub = [[self[i][j]-other[i][j] for j in range(self.w)] for i in range(self.h)]
        return Matrix(matrix_sub)

    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        #   
        # TODO - your code here
        #
        if self.w != other.h:
            raise(ValueError, "number of columns of the pre-matrix must equal the number of rows of the post-matrix")    
        result = []
        row = []
        ele_of_matrix = 0
        for i in range(self.h):
            for j in range(other.w):
                ele_of_matrix = sum([self[i][k]*other.T()[j][k] for k in range(self.w)])
                row.append(ele_of_matrix)
            result.append(row)
            row = []
            
        # the second method
        # result = [[sum([self[i][k]*other.T()[j][k] for k in range(self.w)]) for j in range(other.w)] for i in range(self.h)]
        return Matrix(result)
                                         

    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            pass
            #   
            # TODO - your code here
            #
            result = []
            row = []
            for i in range(self.h):
                for j in range(self.w):
                    row.append(other*self[i][j])
                result.append(row)
                row = []
            return Matrix(result)
            #the second method
            #return [[other*self[i][j] for j in range(self.w)] for i in range(self.h)]
            
            



m = Matrix([[1, 2], 
        [3, 4]])

print(m)

print(type(m))
print(type(m[0]))