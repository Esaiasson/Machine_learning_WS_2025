import matplotlib.pyplot as plt
import pandas as pd 

def plot_parameter(train_results_food_waste, train_results_college_rank, param_grid, eval_param):
    df_food_waste = train_results_food_waste.join(train_results_food_waste['Parameters'].apply(pd.Series))
    df_college_rank = train_results_college_rank.join(train_results_college_rank['Parameters'].apply(pd.Series))
    df_food_waste = df_food_waste.drop("Parameters", axis=1)   
    df_college_rank = df_college_rank.drop("Parameters", axis=1)   
    
    for param in param_grid.keys():

        df_static_food_waste = df_food_waste.copy()
        df_static_college_rank = df_college_rank.copy()
        for key, val in eval_param.items():
            if key != param:
                df_static_food_waste = df_static_food_waste[df_static_food_waste[key] == val]
                df_static_college_rank = df_static_college_rank[df_static_college_rank[key] == val]

        df_static_food_waste = df_static_food_waste.sort_values(by=param)        
        df_static_college_rank = df_static_college_rank.sort_values(by=param)        


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{param}")

        #Runtime plot
        ax1.plot(df_static_food_waste[param], df_static_food_waste["runtime"])
        ax1.plot(df_static_college_rank[param], df_static_college_rank["runtime"])
        ax1.set_xlabel(f"{param}")
        ax1.set_ylabel("Runtime")
        
        #Performance metric plot
        ax2.plot(df_static_food_waste[param], df_static_food_waste["mean_rmse"])
        ax2.plot(df_static_college_rank[param], df_static_college_rank["mean_rmse"])
        ax2.set_xlabel(f"{param}")
        ax2.set_ylabel("Score")
        

        plt.tight_layout()
        plt.show()