from helper import adv_search
import random
import numpy as np

class ga(adv_search):
  def __init__(self, f, bounds, cross_over_p=0.3,mutate_p=0.05,delta_norm=0.01,):
    super().__init__(f, bounds, delta_norm)
    self.cross_over_p=cross_over_p
    self.mutate_p=mutate_p

  def __str__(self):
    return "Genetic Algorithm"
  
  def tournament_selection(self,fitness_vals,k):
    indices=random.sample(range(len(fitness_vals)),k=k)
    min_index=indices[0]
    for i in range(k):
      if fitness_vals[indices[i]]<fitness_vals[min_index]:
        min_index=indices[i]
    
    return min_index
  
  def cross_over(self,agent1,agent2):
    new_agent1=agent1.copy()
    new_agent2=agent2.copy()

    for i in range(len(self.bounds)):
      if random.random()<self.cross_over_p:
        alpha=random.random()
        new_agent1[i]=agent1[i]*alpha+agent2[i]*(1-alpha)
        new_agent2[i]=agent1[i]*(1-alpha)+agent2[i]*(alpha)
      
    return new_agent1,new_agent2
  


  def mutate(self,x,cur_iter,max_iter):
    #formula: xnew = x + tau*(xmax - xmin )(1 - r**((1 - t/tmax)^b) )
    if random.random()<0.5:
      tau=1
    else:
      tau=-1
    ans=x.copy()
    for i in range(len(self.bounds)):
      if random.random()<self.mutate_p: # mutate
        ans[i]+=tau*(self.bounds[i][1]-self.bounds[i][0])*(1-random.random()**(1-cur_iter/max_iter))
    
    ans=self.clip(ans.reshape(1,-1)).reshape(-1)

    return ans


  def search(self,max_iter,num_agent,num_cross,tourn_k=4):
    self._log=[]
    
    #random initialization
    population=np.zeros((num_agent,len(self.bounds)))
    for i in range(num_agent):
      population[i]=self.random_init()
    fitness_vals=[self.f(population[i]) for i in range(num_agent)]
    
    population=population[np.argsort(fitness_vals)]
    fitness_vals=sorted(fitness_vals)
    
    for iter in range(1,max_iter+1):
      #fiddling around
      self._log.append(fitness_vals)

      #select and crossover
      children_population=np.zeros((2*num_cross,len(self.bounds)))
      for i in range(num_cross):
        agent1=population[self.tournament_selection(fitness_vals,tourn_k)]
        agent2=population[self.tournament_selection(fitness_vals,tourn_k)]

        child1,child2=self.cross_over(agent1,agent2)
        children_population[2*i]=child1
        children_population[2*i+1]=child2
      
      #mutate
      for i in range(children_population.shape[0]):
        children_population[i]=self.mutate(children_population[i],iter,max_iter)

      #apply elitism to choose next batch
      aug_pop=np.vstack((population,children_population))
      aug_fitness=[self.f(aug_pop[i]) for i in range(aug_pop.shape[0])]
      arg_sorted_aug_fitness=np.argsort(aug_fitness)

      #setting population and fitness
      population=aug_pop[arg_sorted_aug_fitness[:num_agent]]
      fitness_vals=sorted(aug_fitness)[:num_agent]

    return self.f(population[0]),population[0]



