# working with  array  with numpy
import  numpy as np

arr = np. array([1, 2, 3, 4, 6])

print(arr)




import numpy as np

arr = np.array((1, 2, 3, 4, 5)) # use of tuple

print(arr)
print(type(arr))



#Create a 0-D array with value 42

import numpy as np

arr = np.array(4)

print(arr)
print(arr.ndim)   # which dimension array it is...
#Create a 1-D array containing the values 1,2,3,4,5:

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(arr.ndim)   # which dimension array it is...

#Create a 2-D array containing two arrays with the values 1,2,3 and 4,5,6:
         
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)
print(arr.ndim)     # which dimension array it is...


#Create a 3-D array  FROM 2D containing two arrays with the values 1,2,3 and 
#4,5,6 ,.....:

import numpy as np

a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])


print(a)
print(a.ndim)    # which dimension array it is...
# NOTICE THE , IS USED FOR NEXT LINE  AND [ ] IS USED FOR WHICH TYPE OF 
# DIMENSION YOU WANT SIMPLY NUMBER OF [ ] SIGNIFIES NUMBER OF DIMENSIONS

import numpy as np
a=([["rp","jatt",],["ratinder" ,"rp"]])
print(a)
print(arr.ndim)