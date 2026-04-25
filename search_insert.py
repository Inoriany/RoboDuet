# class Solution:
#     def searchInsert(self, nums: List[int], target: int) -> int:
#         mid = len(nums) // 2
#         if nums[mid] < target:
#             return self.searchInsert(nums[mid + 1:], target)
#         elif nums[mid] > target:
#             return self.searchInsert(nums[:mid], target)
#         elif nums[mid] == target:
#             return mid

        
# 别急，你原来的思路其实是对的（二分查找），就是有几个小坑。一个一个看：

# 坑1：没写 return

# # 你的
# if nums[mid] < target:
#     searchInsert(nums[mid + 1:], target)    # 结果算了但丢了

# # 加上return
# if nums[mid] < target:
#     return self.searchInsert(nums[mid + 1:], target)
# 就像你问别人帮你查一个答案，别人告诉你了，但你没听，直接走了。

# 坑2：没写 self.

# # 你的
# searchInsert(nums[mid + 1:], target)    # 找不到这个函数

# # 因为是class里的方法，要加self
# self.searchInsert(nums[mid + 1:], target)
# 坑3：切片后索引对不上（最致命）

# nums = [1, 3, 5, 7, 9],  target = 7

# mid=2, nums[2]=5 < 7
# 传入 nums[3:] → [7, 9]

# 在 [7, 9] 里找到7，mid=0，返回 0
# 但7在原数组里是 index 3！
# 切片创建了新数组，索引从0重新开始，和原数组对不上了。

# 坑4：没处理找不到的情况

# nums = [1, 3, 5], target = 4

# 最终会递归到 nums=[] 空数组
# nums[mid] → 报错！没有元素可以取
# 所以用左右指针的写法更好，不切片，不递归，没有这些坑。


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] <= target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
        return left

        