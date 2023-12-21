from hill_climbing import hill_climber
from sim_annealing import sim_annealing
from ga import ga
from helper import func1,fifty_d_rosenbrock
import sys

if __name__=="__main__":

  folder_name="./result/"
  functions=[
    func1(),
    fifty_d_rosenbrock(4),
  ]
  for func in functions:
    algorithms=[
      hill_climber(func,func.get_bounds(),0.01),
      sim_annealing(func,func.get_bounds(),0.1),
      ga(func,func.get_bounds()),
    ]
    parameters=[
      (20,5),
      (100,),
      (100,20,5,4),
    ]
    for idx in range(3):
      alg=algorithms[idx]
      par=parameters[idx]    
      best_fit,best_loc=alg.search(*par)
      
      with open(folder_name+f'{func.__str__()}_{alg.__str__()}.txt','w') as f:
        print(f"optimizing function: {func.__str__()}",file=f)
        alg.dump_log(f)
        print(f"best_value:{best_fit} and location:{best_loc}",file=f)


