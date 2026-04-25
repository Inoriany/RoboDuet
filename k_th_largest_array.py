class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        pivot = random.choice(nums)
        big = [x for x in nums if x > pivot]
        equal = [x for x in nums if x == pivot]
        small = [x for x in nums if x < pivot]

        if k <= len(big):
            return self.findKthLargest(big, k)
        elif k <= len(big) + len(equal):
            return pivot
        else:
            return self.findKthLargest(small, k - len(big) - len(equal))
            
        