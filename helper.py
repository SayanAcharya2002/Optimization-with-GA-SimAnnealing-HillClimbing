import numpy as np
import random

class obj_func_class:
  def __init__(self):
    pass
  def __call__(self):
    pass
  def get_bounds(self):
    pass
  def __str__(self):
    return "base_class_obj_func"
  
class adv_search:
  def __init__(self,f:obj_func_class,bounds,delta_norm):
    self.f=f
    self.bounds=bounds
    self.delta_norm=delta_norm
    self._log=[]
  def __str__(self):
    return "base_class_adv_search_algo"
  def __repr__(self):
    return self.__str__()
  
  def dump_log(self,fp):
    fp.write(f"ALGORITHM: {self.__str__()}\n")
    fp.write("{:10} {:10}\n".format("epoch","best_fit"))
    fp.write("{:10} {:10}\n".format("-"*10,"-"*10))
    for i in range(len(self._log)):
      best_fit=self._log[i]
      if isinstance(self._log[i],list): 
        best_fit=self._log[i][0]
      fp.write("{:<10} {:10}\n".format(i,best_fit))


  def generate_neighbors(self,state,is_random=True):
    l=[]
    for i in range(len(state)):
      delta=(self.bounds[i][1]-self.bounds[i][0])*self.delta_norm
      if is_random:
        delta*=random.random()/2+0.5
      l.append(state.copy())
      l.append(state.copy())
      l[-1][i]+=delta
      l[-2][i]+=-delta
    l=np.array(l)

    l=self.clip(l)

    return l
  
  def clip(self,l):
    #clipping(can try randomizing here too)
    for i in range(len(self.bounds)):
      l[:,i]=np.clip(l[:,i],*self.bounds[i])

    return l

  def random_init(self):
    return np.array([
      mini+(maxi-mini)*random.random() for mini,maxi in self.bounds 
    ])
  
  def search(self):
    pass


class fifty_d_rosenbrock(obj_func_class):
  def __init__(self,n=10):
    super().__init__()
    self.n=n

  def __call__(self,x):
    ans=0
    for i in range(len(x)-1):
      ans+=100*((x[i+1]-x[i]**2)**2)+(1-x[i])**2
    return ans
  
  def get_bounds(self):
    return [(-5,5)]*self.n
  
  def __str__(self):
    return "rosenbrock_function"


class func1(obj_func_class):
  def __init__(self):
    super().__init__()

  def __call__(self,l):
    ans=(l[0]**2)*np.cos(l[1]-l[2]**3)+np.sin(l[1])*l[0]+l[1]*l[3]
    return ans
  
  def get_bounds(self):
    return [
      (0,1),
      (0,10),
      (5,6),
      (1,100),
    ]
    # return [(0,1)]*4

  def __str__(self):
    return "custom_function1"
