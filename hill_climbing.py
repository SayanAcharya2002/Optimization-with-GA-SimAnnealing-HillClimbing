import numpy as np
import random
from helper import adv_search

class hill_climber(adv_search):

  def __init__(self,f,bounds,delta_norm=0.01):
    super().__init__(f,bounds,delta_norm)
  def __str__(self):
    return "hill_climber"

  def generate_neighbors(self,state):
    return super().generate_neighbors(state)

  def random_init(self):
    return super().random_init()
  
  def search(self,k,max_iter):
    self._log=[]
    best_fit=np.inf
    best_loc=None

    for _ in range(k):
      cur_state= self.random_init() #random init
      for iter_count in range(max_iter):
        cur_fit=self.f(cur_state)
        self._log.append(cur_fit)
        # print(f"{iter_count}: {cur_fit}")
        if cur_fit<best_fit:
          best_fit=cur_fit
          best_loc=cur_state.copy()

        neigh=self.generate_neighbors(cur_state)
        obj_list=np.array([self.f(i) for i in neigh])
        obj_arg_sorted=np.argsort(obj_list)

        if obj_list[obj_arg_sorted[0]]<cur_fit:#better neigh exists
          cur_state=neigh[obj_arg_sorted[0]]
        else: #better neigh does not exist
          if random.random()>0.5: # least bad sol
            cur_state=neigh[obj_arg_sorted[0]]
          else: #random neighbor choice
            cur_state=neigh[random.randrange(0,len(neigh))]
        
    return best_fit,best_loc
  
