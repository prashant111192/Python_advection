import numpy as np

#make an array of 
a = np.array([0,1,2,3,4,5,6,7,8,9])
print(a)
print(a[0:3])
print(a[1])
s = 1.782345823 
print(s)
print((int)(s))
ss = (int)(s)
print((str)(ss))
sss = np.array(a, dtype=str)
print(sss)
np.savetxt('test.txt', sss, fmt='%s')






    
    

