import matplotlib.pyplot as plt
import pandas as pd 

def plot_parameter(train_results_food_waste, train_results_college_rank, param_grid, eval_param):
    df_food_waste = train_results_food_waste.join(train_results_food_waste['Parameters'].apply(pd.Series))
    df_college_rank = train_results_college_rank.join(train_results_college_rank['Parameters'].apply(pd.Series))    
  
    
    #print(df_food_waste)
    for param in param_grid.keys():

        df_static_food_waste = df_food_waste.copy()
        df_static_college_rank = df_college_rank.copy()
        for key, val in eval_param.items():
            if key != param:
                df_static_food_waste = df_static_food_waste[df_static_food_waste[key] == val]
                df_static_college_rank = df_static_college_rank[df_static_college_rank[key] == val]


        df_static_food_waste = df_static_food_waste.sort_values(by=param)        
        df_static_college_rank = df_static_college_rank.sort_values(by=param)        


        print(param)
        print(df_static_food_waste[[param, "runtime"]])
        plt.figure(figsize=(8, 6))
        plt.plot(df_static_food_waste[param], df_static_food_waste["runtime"])
        plt.plot(df_static_college_rank[param], df_static_college_rank["runtime"])
        plt.xlabel(f"{param}")
        plt.ylabel("Runtime")
        plt.tight_layout()
        plt.show()