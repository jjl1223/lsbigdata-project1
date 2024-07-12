a=(1,2,3)
a

a=[1,2,3]
a

b=a #강한 복사 a의 변화를 b가 그대로 따라감
b
a[1]=4
a

b

a=[1,2,5]
id(a)

b=a[:] #얕은 복사를 한다는 뜻 이후 a의 변화를 따라가지않음
b=a.copy() #얕은 복사를 한다는 뜻 이후 a의 변화를 따라가지않음

id(b)

a[1]=4
a
b
import math
x=4
math.sqrt(x)

exp_val = math.exp(5)
print('exp_val값은',exp_val)

cos_val=math.cos(3)
print('exp_val값은',cos_val)

def my_normal_pdf(x,mu,sigma):
   part1= 1/(sigma *math.sqrt(2*math.pi)) #(sigma *math.sqrt(2*math.pi))**-1
   part2= math.exp(-(x-mu)**2/2*sigma**2)
   return part1*part2
 import math
 def function(x,y,z):
    my_ensure=(x**2+math.sqrt(y)+math.sin(math.radians(z)))*math.exp(1)**x
    return my_ensure
  x=1 
  y=1 
  z=1
  
  function(x,y,z)
  def sce(x):
    ade=math.sin(x)+math.cos(x)+math.exp(x)
    return ade
x=
def dffdf(input):
contents
return 

    import pandas as pd
        import numpy as np    
    import pandas as pd
    import numpy as np    
    import pandas as pd
