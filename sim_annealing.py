from helper import adv_search
import numpy as np
import random,math

class sim_annealing(adv_search):
  def __init__(self, f, bounds, delta_norm=0.01):
    super().__init__(f,bounds,delta_norm)
  
  def __str__(self):
    return "simulated_annealing"

  def random_init(self):
    return super().random_init()

  def generate_neighbors(self, state):
    return super().generate_neighbors(state)

  def search(self,max_iter:int):
    self._log=[]
    best_fit=np.inf
    best_loc=None
    cur_state=self.random_init()
    temp=1
    mod_coeff=0.99
    for iter in range(max_iter):
      temp=temp*mod_coeff
      neighs=self.generate_neighbors(cur_state)
      rand_neigh=random.choice(neighs)
      cur_fit=self.f(cur_state)
      self._log.append(cur_fit)
      rand_fit=self.f(rand_neigh)
      # print(f"iter_no:{iter} cur_fit:{cur_fit}")
      if(cur_fit<best_fit):
        best_fit=cur_fit
        best_loc=cur_state

      if rand_fit<cur_fit:#invariably accept
        cur_state=rand_neigh
      else:
        if random.random()<np.exp(-math.pow(abs(rand_fit-cur_fit),1/10)/temp): #accept
          cur_state=rand_neigh
        
    return best_fit,best_loc

