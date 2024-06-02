import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

#gurobi
import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr
import gurobipy as gpf


################################# set page configuration #################################
st.set_page_config(layout="wide")



######################## FUNCTIONS USES TO RUN THE APP ########################

@st.cache_data(show_spinner="Loading data...")
def read_historical_data():
    ##### read data that have all the units sold for each region
    path_data_basic_features = 'data/data_basic_features.pkl'
    data_units_sold = pd.read_pickle(path_data_basic_features)
    
    ##### use data to generate parameters for optimization model
    # min, max deliry each region
    data_min_delivery = data_units_sold.groupby("region")["units_sold"].min().rename('min_delivery')
    data_max_delivery = data_units_sold.groupby("region")["units_sold"].max().rename('max_delivery')
    
    # historical distribution of price each region
    data_historical_max_price = data_units_sold.groupby("region")["price"].max().rename('max_price')

    return data_min_delivery, data_max_delivery, data_historical_max_price


@st.cache_resource(show_spinner="Loading models...")
def read_ml_models_trained():
    # params
    path_folder_artifacts = 'models/'
    list_models_names = os.listdir(path_folder_artifacts)
    
    ### load models
    dict_models = {}
    for model_name in list_models_names:
        # params
        #print(f'loading model: {model_name}')
        path_model = path_folder_artifacts + model_name
        
        # load
        aux = model_name.split('.')[0].split('_')[1:]
        model_name_index = '_'.join(aux)
        with open(path_model, 'rb') as artifact:
            dict_models[model_name_index] = pickle.load(artifact)

    return dict_models



######################## ORDER CODES THAT SHOW INFORMATION IN THE UI ########################
if __name__ == "__main__":


    ######################## FORM TO INPUT VALUES OF OPTIMIZER - SIDEBAR ########################
    with st.form(key ='Form1'):
        with st.sidebar:
            st.header('----- INPUT PARAMS TO RUN OPTIMIZATION -----')
            
            ############## SEASONALITY - FIXED INPUT MACHINE LEARNING MODEL ##############
            st.divider()
            st.write('**Seasonalilty: Peak moths - Normal Months**')
            input_seasonality_peak_binary = st.checkbox("Peak months")
            if input_seasonality_peak_binary:
                input_seasonality_peak = 1 # true
            else:
                input_seasonality_peak = 0 # false
            

            ############## PARAMETERS OF OPTIMIZATION PROBLEM ##############
            st.divider()
            col1_sidebar, col2_sidebar, col3_sidebar = st.columns(3)

            ### COLUMN 1 - PRICES MIN AND MAX OF PRODUCT
            col1_sidebar.write('**Range min-max price of the product**')
            input_product_price_min = col1_sidebar.number_input("Min price", min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1) #input_product_price_min = 0
            input_product_price_max = col1_sidebar.number_input("Max price", min_value = 1.0, max_value = 5.0, value = 2.0, step = 0.1) #input_product_price_max = 2


            ### COLUMN 2 - SUPPLY PRODUCT FOR EACH REGIONS
            col2_sidebar.write('**Total supply of product**')
            input_supply_product = col2_sidebar.number_input("Supply Product", min_value = 0, max_value = 40, value = 25, step = 1) # input_supply_product = 25

            ### COLUMN 3 - ### COSTS - TRANSPORT - WASTE - ETC
            col3_sidebar.write('**Costs of transport each region**')

            input_c_waste = col3_sidebar.number_input("cost waste all regions", min_value = 0.0, max_value = 1.5, value = 0.1, step = 0.1) # input_c_waste = 0.1

            input_c_transport_Great_Lakes = col3_sidebar.number_input("Cost Transport Great Lakes", min_value = 0.0, max_value = 1.5, value = 0.3, step = 0.1) # input_c_transport_Great_Lakes = 0.3
            input_c_transport_Midsouth = col3_sidebar.number_input("Cost Transport Midsouth", min_value = 0.0, max_value = 1.5, value = 0.1, step = 0.1) # input_c_transport_Midsouth = 0.1
            input_c_transport_Northeast = col3_sidebar.number_input("cost Transport Northeast", min_value = 0.0, max_value = 1.5, value = 0.4, step = 0.1) # input_c_transport_Northeast = 0.4
            input_c_transport_Northern_New_England = col3_sidebar.number_input("Cost Transport Northern New England", min_value = 0.0, max_value = 1.5, value = 0.5, step = 0.1) # input_c_transport_Northern_New_England = 0.5
            input_c_transport_SouthCentral = col3_sidebar.number_input("Cost Transport SouthCentral", min_value = 0.0, max_value = 1.5, value = 0.3, step = 0.1) # input_c_transport_SouthCentral = 0.3
            input_c_transport_Southeast = col3_sidebar.number_input("Cost Transport Southeast", min_value = 0.0, max_value = 1.5, value = 0.2, step = 0.1) # input_c_transport_Southeast = 0.2
            input_c_transport_West = col3_sidebar.number_input("Cost Transport West", min_value = 0.0, max_value = 1.5, value = 0.2, step = 0.1) # input_c_transport_West = 0.2
            input_c_transport_Plains = col3_sidebar.number_input("Cost Transport Plains", min_value = 0.0, max_value = 1.5, value = 0.2, step = 0.1) # input_c_transport_Plains = 0.2

            ############## SUBMIT BUTTON ##############
            submitted_opt = st.form_submit_button(label = 'Run Optimization')




    ######################## RUN OPTIMIZATION WHEN USER SEND THE NEW VALUES OF OPTIMIZATION ########################
    if submitted_opt:
        ## PREPARARION
        ### 1. Load data needs to use
        data_min_delivery, data_max_delivery, data_historical_max_price = read_historical_data()
        dict_models = read_ml_models_trained()

        ## RUN OPTIMIZER
        ### 0. Load transversal params - sets of optimization model
        list_regions = ['Great_Lakes', 'Midsouth', 'Northeast', 'Northern_New_England', 'Plains', 'SouthCentral', 'Southeast', 'West']
        regions = list_regions
        index_regions = pd.Index(regions)


        ### 1. Create guroby optimization model
        model_opt = gp.Model(name = "Avocado_Price_Allocation")


        ### 2. Upper bounds and lower bounds of decision variables
        # product_price_min, product_price_max: min and max price of product A
        product_price_min = input_product_price_min
        product_price_max = input_product_price_max

        # b_min(r), b_max(r): min and max historical products send to each region (value get from historical data)
        b_min = data_min_delivery
        b_max = data_max_delivery


        ### 3. Input parameters of optimization model
        # B: supply product
        B = input_supply_product

        # c_waste: cost of waste product
        c_waste = input_c_waste

        # c_transport(r): cost transport for each region
        c_transport = pd.Series(
            {
                "Great_Lakes": input_c_transport_Great_Lakes,
                "Midsouth": input_c_transport_Midsouth,
                "Northeast": input_c_transport_Northeast,
                "Northern_New_England": input_c_transport_Northern_New_England,
                "SouthCentral": input_c_transport_SouthCentral,
                "Southeast": input_c_transport_Southeast,
                "West": input_c_transport_West,
                "Plains": input_c_transport_Plains,
            }, name='transport_cost')
        c_transport = c_transport.loc[regions]


        ### 4. Features input machine learning model fixed (that are not decision variables or parameters in optimization model)
        peak_or_not = input_seasonality_peak
        instance_ml_model = pd.DataFrame(
            data={
                "peak": peak_or_not
            },
            index=regions
        )


        ### 5. Decision variables of optimization model
        # p(r): price. feature of machine learning model
        price = gppd.add_vars(model_opt, index_regions, name = "price", lb = product_price_min, ub = product_price_max)

        # x(r): supply
        supply = gppd.add_vars(model_opt, index_regions, name = "supply", lb = b_min, ub= b_max)

        # s(r): solds given a certain price
        sold = gppd.add_vars(model_opt, index_regions, name = "sold")

        # u(r): inventary. units not sold. waste.
        inventory = gppd.add_vars(model_opt, index_regions, name = "inventory") 

        # d(r): demand. output of machine learning model
        demand = gppd.add_vars(model_opt, index_regions, lb = -gp.GRB.INFINITY, name = "demand")


        ### 6. Constraints (constraints that are not generated by a ml model)
        #### 6.1 Add the Supply Constraint
        model_opt.addConstr(supply.sum() == B, name = 'supply')

        #### 6.2 Add Constraints That Define Sales Quantity
        gppd.add_constrs(model_opt, sold, gp.GRB.LESS_EQUAL, supply, name = 'solds <= supply')
        gppd.add_constrs(model_opt, sold, gp.GRB.LESS_EQUAL, demand, name = 'solds <= demand')

        #### 6.3 Add the Wastage Constraints
        gppd.add_constrs(model_opt, inventory, gp.GRB.EQUAL, supply - sold, name = 'waste')

        #### 6.4 Model update - add the constraint to gurobi model
        model_opt.update()


        ### 7. Add constraints that are machine learning models
        for region in regions:

            # there is a dataframe with features fixed (no decision variables). filter it by region
            aux_features_fixed = instance_ml_model.loc[[region]]  
            
            # create a dataframe with decision variables gurobi. filter it by region. In this example the price of all regions are features of the ml model
            aux_features_decision =  pd.DataFrame(price).T
            aux_features_decision.index = [region]
            
            #name_columns_feature_decision = aux_features_decision.columns # CORRECTION NAME COLUMNS TO BE THE SAME COLUMNS NAMES IN DATAFRAME USED TO TRAIN
            name_columns_feature_decision = ['price_' + name_region for name_region in list_regions]
            name_columns_feature_decision = [column.lower() for column in name_columns_feature_decision]
            aux_features_decision.columns = name_columns_feature_decision
            
            # join into a dataframe instance
            instance = pd.concat([aux_features_fixed, aux_features_decision], axis=1) # generate instance
            
            ############ create constraint based in machine learning model ############
            # load model
            model_ml = dict_models[region]
            
            ## add model to predict the demand for each region with the SAME MODEL
            pred_constr = add_predictor_constr(gp_model = model_opt, 
                                            predictor = model_ml, 
                                            input_vars = instance, 
                                            output_vars = demand[region], # filter decision variable for the element of the set region,
                                            name = f'model_predict_{region}'
                                            )


        ### 8. Define Objetive Function
        model_opt.setObjective((price * sold).sum() - c_waste * inventory.sum() - (c_transport * supply).sum(),
                    gp.GRB.MAXIMIZE)


        ### 9. Solve optimization problem
        model_opt.Params.NonConvex = 2
        model_opt.optimize()
        model_status = model_opt.Status # if 2 a optimal solution was founded

        if model_status == 2:
            ### 10. Save optimal values in a dataframe
            solution = pd.DataFrame(index=index_regions)

            # save optimal values
            solution["Price"] = price.gppd.X
            solution["Historical_Max_Price"] = data_historical_max_price  # this is informative value get from historical data
            solution["Allocated(supply)"] = supply.gppd.X
            solution["Sold"] = sold.gppd.X
            solution["Inventory"] = inventory.gppd.X
            solution["Pred_demand"] = demand.gppd.X
            solution["Diff Demand - Supply"] = demand.gppd.X - supply.gppd.X

            # sum values
            total_sum = solution.sum()
            total_sum["Price"] = np.NaN
            total_sum["Historical_Max_Price"] = np.NaN
            solution.loc["Total", :] = total_sum

            # round values
            solution = solution.round(3)


            #### 10.1 show value objetive function
            opt_revenue = model_opt.ObjVal
            opt_revenue = np.round(opt_revenue, 2)

        else:
            pass






    ######################## INFO IN MAIN PAGE ########################
    

    # two tabs - first one show results - second show detail of model
    tab1, tab2 = st.tabs(["Results Optimization", "Details Optimization Modeling"])

    #### COLUMN1

    tab1.markdown("### ----- RESULTS OPTIMIZATION -----")
    if submitted_opt:
        if model_status == 2: # optimal solution was founded
            # show solution
            tab1.write(f"\n The optimal net revenue: ${opt_revenue} million")
            tab1.dataframe(solution)

            # download solution
            csv_solution = solution.to_csv()
            tab1.download_button("Download Optimal Solution", csv_solution, file_name='solution.csv', key='csv_key')
            #os.remove('solution.csv')
        else:
            tab1.write("Model is infeasible or unbounded - Change the input parameters")
    else:
        pass



    #### COLUMN2
    tab2.markdown("### ----- INFO OPTIMIZATION PROBLEM -----")

    tab2.markdown('#### Objetive Function')
    tab2.markdown("Maximize: $\sum_{r} (price_r * sold_r - c_{waste} * inventory_r - c^r_{transport} * supply_r)$") # change the latex code of jupyter notebook for "$"


    tab2.write('\n')
    tab2.write('\n')
    tab2.markdown('#### Constraints')
    tab2.write('\n')
    tab2.markdown('**Supply Constrain**')
    tab2.markdown(r"$\sum_{r} \text{supply}_r = B$")

    tab2.write('\n')
    tab2.markdown('**Sales Quantity**')
    tab2.markdown(r"$\text{sold}_r \leq \text{supply}_r \quad \forall r$")
    tab2.markdown(r"$\text{sold}_r \leq \text{demand}(p_r, r) \quad \forall r$")


    tab2.write('\n')
    tab2.markdown('**Wastage**')
    tab2.markdown(r"$\text{inventory}_r = \text{supply}_r - \text{sold}_r \quad \forall r$")

    tab2.write('\n')
    tab2.markdown('**Demand is function (ml model) of prices**')
    tab2.markdown(r"$\text{demand}(r) = f(\text{prices}, \text{others})$")


    tab2.markdown('')
    tab2.markdown('')















