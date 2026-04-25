from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_to_index = {}
        for index, each in enumerate(nums):
            complement = target - each
            if complement in num_to_index:
                return [num_to_index[complement], index]
            num_to_index[each] = index

two_sum = Solution().twoSum([2, 7, 11, 15], 9)
print(two_sum)