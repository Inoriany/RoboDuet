from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        for i in range(len(prices)-1):
            for j in range(i+1, len(prices)):
                profit = prices[j] - prices[i]
                max_profit = max(max_profit, profit)
        return max_profit
    def maxProfit2(self, prices: List[int]) -> int:
        min_price = float("inf")
        max_profit = 0
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit
max_profit = Solution().maxProfit([7,1,5,3,6,4])
max_profit2 = Solution().maxProfit2([7,1,5,3,6,4])
print(max_profit)
print(max_profit2)