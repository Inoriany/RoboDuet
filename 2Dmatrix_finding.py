# class Solution:
#     def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
#         rows = len(matrix)
#         cols = len(matrix[0])
#         for i in range(rows):
#             if matrix[i][cols - 1] > target:
#                 left, right = 0, cols - 1   
#                 while left <= right:
#                     mid = (left + right) // 2
#                     if matrix[i][mid] == target:
#                         return True
#                     elif matrix[i][mid] > target:
#                         right = mid - 1
#                     elif matrix[i][mid] < target:
#                         left = mid + 1
#                 return False
#         return False

#记得一定要matrix[i][mid], 不能直接写mid，那是一个索引
#记得一定要left, right = 0, cols - 1放在循环里面，否则mid没有更新

        
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows): 
            if matrix[i][cols - 1] >= target:
                left, right = 0, cols - 1   
                while left <= right:
                    mid = (left + right) // 2
                    if matrix[i][mid] == target:
                        return True
                    elif matrix[i][mid] > target:
                        right = mid - 1
                    elif matrix[i][mid] < target:
                        left = mid + 1
                return False
        return False


        