# -*- coding: utf-8 -*-
"""

 Objective: Develop an algorithm to address the following sequence number generator: 
     
     A sequence of n numbers is considered valid if the sequence begins with 1, ends with
     a given number j, and no two adjacent numbers are the same. Sequences may use any integers
     between 1 and a given number k, inclusive (also 1<=j<=k). Given parameters n, j, and k, 
     count the number of valid sequences. The number of valid sequences may be very large, 
     so express your answer modulo 1010+7.

 Author: Zhe Charles Zhou
 
"""

k = 2281 # change this for upper bound of the range of integers that can be used
n = 347 # change this for the number of digits in the sequence
j = 1 # change this for the last digit number

num = (k-2)*(k-1)

for i in range(5,n) :
   
    if i%2 == 0:
        num = num*(k-1)-(k-1)
    else:
        num = num*(k-1)+(k-1)

if j != 1:
    num=num-1


num%(10^10+7)       