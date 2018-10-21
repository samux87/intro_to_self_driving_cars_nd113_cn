"""
M.intersections，字典，节点坐标

print(map_10.intersections)
{0: [0.7798606835438107, 0.6922727646627362],
 1: [0.7647837074641568, 0.3252670836724646],
 2: [0.7155217893995438, 0.20026498027300055],
 3: [0.7076566826610747, 0.3278339270610988],
 4: [0.8325506249953353, 0.02310946309985762],
 5: [0.49016747075266875, 0.5464878695400415],
 6: [0.8820353070895344, 0.6791919587749445],
 7: [0.46247219371675075, 0.6258061621642713],
 8: [0.11622158839385677, 0.11236327488812581],
 9: [0.1285377678230034, 0.3285840695698353]}



M.roads，列表，相邻节点

print(map_10.roads)
[[7, 6, 5],
 [4, 3, 2],
 [4, 3, 1],
 [5, 4, 1, 2],
 [1, 2, 3],
 [7, 0, 3],
 [0],
 [0, 5],
 [9],
 [8]]

"""

import numpy as np
def shortest_path(M,start,goal):
    # 已经探索的节点，集合
    explored = set()

    # 边界节点，即正在探索的节点，集合
    # 初始状态只有start节点
    forntier = {start}

    # 父节点，保存父子关系，字典
    parents = {} 

    # 记录从起始节点到某个节点的花销，字典
    g = {} 
    # 初始状态下，从起始点到任何节点的花销记作无穷大
    # 从起始点到自身的花销记作0
    infinite = float("inf")
    for each in M.intersections:
        g[each] = infinite
    g[start] = 0
    
    # 启发式f = g + h，字典，从初始节点到某个节点的花销g，
    # 加上预测花销h
    f = {}
    f[start] = h(start, goal, M)
    
    # 测试代码
    # print("f = ", f)
    # print("forntier = ", forntier)

    while forntier:
        # 在字典f中，花销最小的节点
        current = lowest_cost_node(f, explored)
        if current == goal:
            return path(parents, current)

        # 测试代码
        # print("current = ", current)
        # print("边界= ", forntier)
        # print("explored = ", explored)   
        # print("f = ", f)
        # print()         
        

        # 把current节点从边界删除
        forntier.remove(current)

        # 把current加入已探索
        explored.add(current)
        
        # 测试代码
        #print("边界= ", forntier)
        #print("explored = ", explored)        
        #print()      

        # 查找current的相邻节点，
        # 如果该节点既不属于边界，也不属于已探索，
        # 则把该节点加入边界
        for each in M.roads[current]:
            if each not in explored and each not in forntier:
                forntier.add(each)

            # 计算从起点到相邻节点的花销
            new_cost = g[current] + h(current, each, M)
            # 如果通过这条路径到相邻接点，
            # 比通过其他已有路径到相同的节点更短，
            # 记录花销更小的路径，包括父子关系parents，花销，和启发式

            if new_cost < g[each]:

                # 更新父节点
                parents[each] = current

                # 更新从起始点到相邻接点的花销
                g[each] = new_cost

                # 更新启发式f = g + h
                f[each] = g[each] + h(each, goal, M) 
    return  False 

# 
def lowest_cost_node(f, explored):
    lowest_cost = float("inf")
    lowest_cost_node = None
    for each in f:
        cost = f[each]
        # 这一句判断很重要，去除已经处理过的节点，
        # 比如初始节点start到本身的距离是0，
        # 如果不删除，返回的节点将始终是它
        if cost < lowest_cost and each not in explored:
            lowest_cost = cost
            lowest_cost_node = each
    return lowest_cost_node

def h(a, b, M):
    return np.sqrt((M.intersections[a][0]-M.intersections[b][0])**2 + \
    (M.intersections[a][1]-M.intersections[b][1])**2)

# 输入父子关系的字典，和目标节点，输出路径（列表）
# 注意：得到的路径可能是倒序的，需要转换一下顺序！

# 代码审阅：因为你在遍历节点的过程中记录了每个节点的parent信息，
# 所以在恢复路径的时候只能从终点到起点回溯，
# 这并不会影响时间复杂度（O(n),大多是学员也是这么做的）.
# 如果你一定不想做翻转操作，就要在遍历过程中记录子节点信息。
def path(parents, current):
    total_path = [current]
    while current in parents:
        current = parents[current]
        total_path.append(current)
    total_path.reverse()
    return total_path