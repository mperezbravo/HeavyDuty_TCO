import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy_financial as npf
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import nbformat
import colormaps
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Create the path to the results folder
results_folder = "../results"  # Assuming this is run from the 'src' folder

# Ensure the results directory exists
os.makedirs(results_folder, exist_ok=True)


def Create_Waterfall(Cases,Case,y, Total_Matrix, data, CONSTANTS):
    Components=["CAPEX", "FIXOM", "VAROM", "Energy", "ETS", "Re-Infr"]
    Technologies_2=["Diesel", "HVO", "NG", "BG", "BET", "FCET"]
    c=Case
    v= Case[1]["VECTO_Group"] 
    import plotly.graph_objects as go
    import nbformat
    import colormaps
    import seaborn as sns
    colors=sns.color_palette("colorblind").as_hex()
    color_diesel=CONSTANTS["color_diesel"]
    color_LNG=CONSTANTS["color_LNG"]
    color_HVO=CONSTANTS["color_HVO"]
    color_bioLNG=CONSTANTS["color_bioLNG"]
    color_BET=CONSTANTS["color_BET"]
    color_FCEV=CONSTANTS["color_FCEV"]


    n_cuts=10
    palette_diesel=sns.dark_palette(color_diesel,n_cuts, reverse=True).as_hex()[0:7]
    palette_LNG=sns.dark_palette(color_LNG,n_cuts, reverse=True).as_hex()[0:7]
    palette_bioLNG=sns.dark_palette(color_bioLNG,n_cuts, reverse=True).as_hex()[0:7]
    palette_HVO=sns.dark_palette(color_HVO,n_cuts, reverse=True).as_hex()[0:7]
    palette_BET=sns.dark_palette(color_BET,n_cuts, reverse=True).as_hex()[0:7]
    palette_FCEV=sns.dark_palette(color_FCEV,n_cuts, reverse=True).as_hex()[0:7]

    colores=np.concatenate((palette_diesel, palette_HVO, palette_LNG, palette_bioLNG, palette_BET, palette_FCEV), axis=None)

    names =[['DIESEL','DIESEL','DIESEL','DIESEL','DIESEL','DIESEL','DIESEL','HVO','HVO','HVO','HVO','HVO','HVO','HVO','LNG','LNG','LNG','LNG','LNG','LNG','LNG','BIO-LNG','BIO-LNG','BIO-LNG','BIO-LNG','BIO-LNG','BIO-LNG','BIO-LNG','BEV','BEV','BEV','BEV','BEV','BEV','BEV','FCEV','FCEV','FCEV','FCEV','FCEV','FCEV', 'FCEV'],['CAPEX','FIXOM','VAROM','ENERGY','ETS','RE-INFR','TOTAL','CAPEX','FIXOM','VAROM','ENERGY','ETS','RE-INFR','TOTAL','CAPEX','FIXOM','VAROM','ENERGY','ETS','RE-INFR','TOTAL','CAPEX','FIXOM','VAROM','ENERGY','ETS','RE-INFR','TOTAL','CAPEX','FIXOM','VAROM','ENERGY','ETS','RE-INFR','TOTAL','CAPEX','FIXOM','VAROM','ENERGY','ETS','RE-INFR','TOTAL','CAPEX','FIXOM','VAROM','ENERGY','ETS','RE-INFR','TOTAL']]
    y_vector=[]

    for t in Technologies_2:
        #print(t)
        for i in Components:
            #print(i)
            y_vector.append(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="Average") & (Total_Matrix["Component"]==i)]["Value"].iat[0])

        y_vector.append(sum(Total_Matrix[(Total_Matrix["Component"].isin(Components)) & (Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="Average")]["Value"]))

    n_components=len(Components)+1
    n_tecs=len(Technologies_2)
    base=[]
    for j in np.arange(0,n_tecs,1):
        base.append(0)
        for k in np.arange(1,(n_components-1)):
            a=sum(y_vector[0+j*n_components:k+j*n_components])
            base.append(a)
        base.append(0)


    minimos=[]
    for t in Technologies_2:
        for i in Components:
            if (len(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="Low") & (Total_Matrix["Component"]==i)]["Value"])>0):
                min=((Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="Low") & (Total_Matrix["Component"]==i)]["Value"].iat[0]))
                ave=((Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="Average") & (Total_Matrix["Component"]==i)]["Value"].iat[0]))
                #print("tech", t)
                #print("comp", i)
                #print("min", min)
                #print("ave", ave)
                #print("app", ave-min)
                minimos.append(ave-min)
                #print(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["VECTO_Group"]==v) & (Total_Matrix["Scenario"]=="Average") & (Total_Matrix["Component"]==i)]["Value"].iat[0]-Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["VECTO_Group"]==v) & (Total_Matrix["Scenario"]=="Low") & (Total_Matrix["Component"]==i)]["Value"].iat[0])
            else:
                minimos.append(0)
        min=sum(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["VECTO_Group"]==v) & (Total_Matrix["Scenario"]=="Low") & (Total_Matrix["Component"].isin(Components) & (Total_Matrix["Case"]==c[1]["Case"]))]["Value"])
        aver=sum(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["VECTO_Group"]==v) & (Total_Matrix["Scenario"]=="Average") & (Total_Matrix["Component"].isin(Components))& (Total_Matrix["Case"]==c[1]["Case"])]["Value"])
        minimos.append(aver-min)

    maximos=[]
    for t in Technologies_2:
        for i in Components:
            if (len(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="Low") & (Total_Matrix["Component"]==i)]["Value"])>0):
                max=((Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="High") & (Total_Matrix["Component"]==i)]["Value"].iat[0]))
                ave=((Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["Case"]==c[1]["Case"]) & (Total_Matrix["Scenario"]=="Average") & (Total_Matrix["Component"]==i)]["Value"].iat[0]))
                #print("tech", t)
                #print("comp", i)
                #print("max", max)
                #print("ave", ave)
                #print("app", max-ave)
                maximos.append(max-ave)
            else:
                maximos.append(0)
        max=sum(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["VECTO_Group"]==v) & (Total_Matrix["Scenario"]=="High") & (Total_Matrix["Component"].isin(Components))& (Total_Matrix["Case"]==c[1]["Case"])]["Value"])
        aver=sum(Total_Matrix[(Total_Matrix["Technology"]==t) & (Total_Matrix["Year"]==y) & (Total_Matrix["VECTO_Group"]==v) & (Total_Matrix["Scenario"]=="Average") & (Total_Matrix["Component"].isin(Components))& (Total_Matrix["Case"]==c[1]["Case"])]["Value"])
        maximos.append(max-aver)

    fig = go.Figure(data=[
        go.Bar(name='', x=names, y=base, marker=dict(color='rgba(0,0,0,0)', line=dict(color='rgba(0,0,0,0)'))),
        go.Bar(name='TCO', x=names, y=y_vector, error_y=dict(type='data',symmetric=False,array=maximos,arrayminus=minimos), marker=go.bar.Marker(color=[f'rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.6)' for c in colores], line=dict(color='black')))        
        #go.Bar(name='TCO', x=x, y=y_vector, error_y=dict(type='data',symmetric=False,array=maximos,arrayminus=minimos), marker=go.bar.Marker(color=colores))
    ])

    fig.update_layout(barmode='stack', yaxis_title="EUR/km")
    fig.update_layout(showlegend=False)
    fig.update_layout(
                    yaxis=dict(
                        tickmode='array',
                        tickvals=np.arange(np.min(y_vector), np.max(y_vector) + 0.1, 0.1),  # Set ticks every 0.1
                        ticktext=[f"{tick:.1f}" for tick in np.arange(np.min(y_vector), np.max(y_vector) + 0.1, 0.1)]
                    ),
                    yaxis_title="EUR/v.km",
                    width=500,
                    height=200,
                    font=dict(
                        size=6,  # Set the font size here
                        color="Black"
                    ),
                    font_family="CMU Serif",
                    margin=dict(l=0, r=0, t=0, b=0),
                                        plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'

                )
    #fig.add_hline(y=y_vector[5],line_width=1, line_dash="dash", line_color="brown")
    
    fig
    filename = "TCO_{}_{}.svg".format(v, y)
    #fig.write_image(filename)
    #print(base)
    #print(y_vector)
    #print(maximos)
    #print(minimos)
    fig.add_hline(y=y_vector[6], line_width=1, line_dash="dash", line_color="brown")
    #fig.write_image(os.path.join(results_folder, filename), format='svg')
    return fig, base, y_vector, maximos, minimos

def plot_tco_evolution(Total_Matrix):
    # Filter data
    df = Total_Matrix[Total_Matrix['Technology'].isin(['NG', 'BG', 'BET', 'FCET', 'Diesel', 'HVO'])]
    Components = ["CAPEX", "FIXOM", "VAROM", "Energy", "ETS", "Re-Infr"]
    scenarios_of_interest = ['Low', 'High', 'Average']
    
    filtered_df = df[df['Scenario'].isin(scenarios_of_interest) & df['Component'].isin(Components)]
    grouped_df = filtered_df.groupby(['Case', 'Technology', 'Year', 'Scenario'])['Value'].sum().reset_index()
    
    # Pivot table
    pivot_df = grouped_df.pivot(index=['Case', 'Technology', 'Year'], columns='Scenario', values='Value').reset_index()
    pivot_df.rename(columns={'Low': 'Low_Value', 'High': 'High_Value', 'Average': 'Average_Value'}, inplace=True)
    
    # Normalize values
    diesel_df = pivot_df[pivot_df['Technology'] == 'Diesel'][['Case', 'Year', 'Average_Value']]
    pivot_df = pivot_df.merge(diesel_df, on=['Case', 'Year'], suffixes=('', '_Diesel'))
    pivot_df.drop(columns=['Average_Value_Diesel'], inplace=True)
    
    # Define order and settings
    case_order = ['Urban Delivery', 'Regional', 'Long Haul - Depot', 'Long Haul - Long Distance']
    selected_years = [2025, 2030, 2035, 2040]
    offset = 0.2
    palette_2 = sns.color_palette("deep")
    colors = {'Diesel': palette_2[0], 'BG': palette_2[1], 'BET': palette_2[2], 'FCET': palette_2[8], 'NG': palette_2[3], 'HVO': palette_2[9]}
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()
    
    for idx, case in enumerate(case_order):
        case_df = pivot_df[pivot_df['Case'] == case]
        ax = axes[idx]
        
        for i, technology in enumerate(case_df['Technology'].unique()):
            tech_df = case_df[case_df['Technology'] == technology]
            tech_df = tech_df[tech_df['Year'].isin(selected_years)]
            
            lower_error = tech_df['Average_Value'] - tech_df['Low_Value']
            upper_error = tech_df['High_Value'] - tech_df['Average_Value']
            errors = [lower_error, upper_error]
            
            jittered_years = tech_df['Year'] + (i - len(case_df['Technology'].unique()) / 2) * offset
            line_color = colors[technology]
            line_style = '--' if technology in ['Diesel', 'NG', 'BG', 'HVO'] else '-'
            lighter_color = to_rgba(line_color, alpha=0.2)
            
            if technology in ['BET', 'FCET']:
                ax.plot(jittered_years, tech_df['Average_Value'], label=technology, marker='o', color=line_color, linestyle=line_style)
                ax.fill_between(jittered_years, tech_df['Low_Value'], tech_df['High_Value'], color=lighter_color, alpha=0.2)
            else:
                ax.errorbar(jittered_years, tech_df['Average_Value'], yerr=errors, label=technology, fmt='-o', capsize=4, ecolor=line_color, color=line_color, linestyle=line_style)
        
        ax.set_title(f'{case}')
        ax.set_xlabel('Year')
        ax.set_ylabel('EUR/v.km')
        ax.set_xticks(selected_years)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, +0.0))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(results_folder, 'TCO_Evolution.svg'), format='svg')
    plt.show()


def plot_tco_fleet_sensitivity(Total_Matrix, y, CONSTANTS):

    # Filter data for the year y
    df_y = Total_Matrix[Total_Matrix["Year"] == y]

    # Define cases and technology-scenario combinations
    cases = df_y["Case"].unique()
    tech_scenarios = [("FCET", f"F_{i}") for i in range(1, 100)]
    x_positions = np.arange(1, 100, 1)  # X-axis positions (1, 2, ..., 99)

    # Create subplots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))  # A4 aspect ratio (landscape)
    legend_handles = []

    # Iterate over cases and corresponding subplot positions
    for i, case in enumerate(cases):
        ax = axes[i // 2, i % 2]  # Convert index to 2D position
        df_case = df_y[df_y["Case"] == case]

        # Compute BET and Diesel benchmark values
        bet_value = df_case[(df_case["Technology"] == "BET") & 
                            (df_case["Scenario"] == "Average") & 
                            (df_case["Component"] != "Re-Infr")]["Value"].sum()
        diesel_value = df_case[(df_case["Technology"] == "Diesel") & 
                               (df_case["Scenario"] == "Average") & 
                               (df_case["Component"] != "Re-Infr")]["Value"].sum()

        # Compute Low and High values for BET and Diesel
        bet_low = df_case[(df_case["Technology"] == "BET") & 
                          (df_case["Scenario"] == "Low") & 
                          (df_case["Component"] != "Re-Infr")]["Value"].sum()
        bet_high = df_case[(df_case["Technology"] == "BET") & 
                           (df_case["Scenario"] == "High") & 
                           (df_case["Component"] != "Re-Infr")]["Value"].sum()
        
        diesel_low = df_case[(df_case["Technology"] == "Diesel") & 
                             (df_case["Scenario"] == "Low") & 
                             (df_case["Component"] != "Re-Infr")]["Value"].sum()
        diesel_high = df_case[(df_case["Technology"] == "Diesel") & 
                              (df_case["Scenario"] == "High") & 
                              (df_case["Component"] != "Re-Infr")]["Value"].sum()

        values, lower_errors, upper_errors = [], [], []
        for tech, scen in tech_scenarios:
            other_components_avg = df_case[(df_case["Technology"] == tech) & 
                                           (df_case["Scenario"] == "Average") & 
                                           (df_case["Component"] != "Re-Infr")]["Value"].sum()
            reinfr_value_avg = df_case[(df_case["Technology"] == tech) & 
                                       (df_case["Component"] == "Re-Infr") & 
                                       (df_case["Scenario"] == scen)]["Value"].sum()
            total_value_avg = other_components_avg + reinfr_value_avg

            other_components_low = df_case[(df_case["Technology"] == tech) & 
                                           (df_case["Scenario"] == "Low") & 
                                           (df_case["Component"] != "Re-Infr")]["Value"].sum()
            other_components_high = df_case[(df_case["Technology"] == tech) & 
                                            (df_case["Scenario"] == "High") & 
                                            (df_case["Component"] != "Re-Infr")]["Value"].sum()

            values.append(total_value_avg)
            lower_errors.append(other_components_avg - other_components_low)
            upper_errors.append(other_components_high - other_components_avg)

        values, lower_errors, upper_errors = np.array(values), np.array(lower_errors), np.array(upper_errors)
        ax.plot(x_positions, values, color=CONSTANTS["color_FCEV"], label="FCET")
        ax.fill_between(x_positions, values - lower_errors, values + upper_errors, color=CONSTANTS["color_FCEV"], alpha=0.4, hatch="//", edgecolor=CONSTANTS["color_FCEV"])
        
        ax.axhline(y=bet_value, linestyle="dotted", color=CONSTANTS["color_BET"], linewidth=1.5, label="BET (Avg)", alpha=0.6)
        ax.fill_between(x_positions, bet_low, bet_high, color=CONSTANTS["color_BET"], alpha=0.2)
        
        ax.axhline(y=diesel_value, linestyle="dashed", color=CONSTANTS["color_diesel"], linewidth=1.5, label="Diesel (Avg)", alpha=0.6)
        ax.fill_between(x_positions, diesel_low, diesel_high, color=CONSTANTS["color_diesel"], alpha=0.2)

        ax.set_title(case, fontsize=14)
        ax.set_ylabel("TCO (EUR/v.km)", fontsize=12)
        ax.set_xlabel("Fleet size")
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xticks(np.arange(0, 50, 10))
        ax.set_ylim(0, max(values + upper_errors) * 1.1)
        ax.set_xlim(0, 50)

        if i == 0:
            legend_handles.extend(ax.lines)

    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    legend_handles = [
        mpatches.Patch(facecolor=CONSTANTS["color_FCEV"], alpha=0.4, hatch='//', edgecolor=CONSTANTS["color_FCEV"], label='FCET'),
        mpatches.Patch(facecolor=CONSTANTS["color_BET"], alpha=0.2, label='BET'),
        mpatches.Patch(facecolor=CONSTANTS["color_diesel"], alpha=0.2, label='Diesel')
    ]
    fig.legend(handles=legend_handles, loc='lower center', fontsize=12, ncol=3, frameon=False, bbox_to_anchor=(0.5, 0))
    plt.savefig(os.path.join(results_folder, 'Fleet_Size_LinePlot_with_ShadedArea_2.png'), format='png')
    plt.show()