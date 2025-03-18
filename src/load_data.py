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


#Components prices interpolation
def add_and_interpolate_all(df):
    # Create a complete range of years until 2050
    all_years = np.arange(df["Year"].min(), 2051)
    
    # Create an empty DataFrame to store the results
    interpolated_df = pd.DataFrame()
    
    # Group by Component and Scenario and interpolate missing years
    for (component, scenario), group in df.groupby(["Component", "Scenario"]):
        # Ensure unique years by dropping duplicates
        group = group.drop_duplicates(subset="Year")
        
        # Reindex the group to include all years until 2050
        group = group.set_index("Year").reindex(all_years).reset_index()
        
        # Interpolate the missing prices
        group["Price"] = group["Price"].interpolate()
        
        # Forward fill to keep constant values until the end
        group["Price"] = group["Price"].ffill().bfill()
        
        # Add the component and scenario columns back
        group["Component"] = component
        group["Scenario"] = scenario
        
        # Append the interpolated group to the result DataFrame
        interpolated_df = pd.concat([interpolated_df, group], ignore_index=True)
    
    return interpolated_df


def load_data(CONSTANTS):
        # Define the base path
    base_path = "../data/"

    # Load datasets with the correct relative paths
    Chassis_Price = pd.read_csv(f"{base_path}data_Chassis.csv")
    Components_Price = pd.read_csv(f"{base_path}data_Components_Prices.csv")
    Efficiencies = pd.read_csv(f"{base_path}data_Efficiencies.csv")
    CAPEX_Vehicles = pd.read_csv(f"{base_path}data_CAPEX_Vehicles.csv")
    Powertrain_Features = pd.read_csv(f"{base_path}data_Powertrain_Features.csv")

    Maintenance_curve = pd.read_csv(f"{base_path}data_Maintenance_Curve.csv")
    Maintenance_Price = pd.read_csv(f"{base_path}data_Maintenance_Price.csv")

    Em_Factors = pd.read_csv(f"{base_path}data_FE_Fuels.csv")
    ETS_Price = pd.read_csv(f"{base_path}data_ETS_Prices.csv")
    Comp_Features = pd.read_csv(f"{base_path}data_Comp_Features.csv")
    Cost_Red_H2 = pd.read_csv(f"{base_path}data_Cost_Red_H2.csv")

    Electrolyzer_IEA= pd.read_csv(f"{base_path}data_Electrolyzer.csv")

    VECTO = pd.read_csv(f"{base_path}data_VECTO.csv")
    VECTO_Groups = pd.Series(VECTO["VECTO_Group"]).drop_duplicates().tolist()

    # Load FuelPrices (assuming it was missing in your original code)
    FuelPrices = pd.read_csv(f"{base_path}data_FuelPrices.csv")  
    # Drop rows that are entirely NA
    FuelPrices = FuelPrices.dropna(how='all')

    # Drop columns that are entirely NA
    FuelPrices = FuelPrices.dropna(axis=1, how='all')

    #Pre-Processing Fuel Prices

    #Pre-Processing efficiencies
    Efficiencies["Efficiency"] = pd.to_numeric(Efficiencies["Efficiency"], errors='coerce')
    Efficiencies_diesel=(Efficiencies["Efficiency"]*CONSTANTS["kWh_l_Diesel"])/100
    Efficiencies_gas=(Efficiencies["Efficiency"]*CONSTANTS["kWh_kg_NG"])/100
    Efficiencies_BG=(Efficiencies["Efficiency"]*CONSTANTS["kWh_kg_NG"])/100
    Efficiencies_H2=(Efficiencies["Efficiency"]*CONSTANTS["kWh_kg_H2"])/100
    Efficiencies_BET=(Efficiencies["Efficiency"])

    Efficiencies["Efficiency_kWh"] = np.nan  # Initialize with NaN
    Efficiencies.loc[Efficiencies.Technology == "Diesel", "Efficiency_kWh"] = Efficiencies_diesel
    Efficiencies.loc[Efficiencies.Technology == "NG", "Efficiency_kWh"] = Efficiencies_gas
    Efficiencies.loc[Efficiencies.Technology == "BG", "Efficiency_kWh"] = Efficiencies_BG
    Efficiencies.loc[Efficiencies.Technology == "BET", "Efficiency_kWh"] = Efficiencies_BET
    Efficiencies.loc[Efficiencies.Technology == "FCET", "Efficiency_kWh"] = Efficiencies_H2
    Efficiencies.loc[Efficiencies.Technology == "HVO", "Efficiency_kWh"] = Efficiencies_diesel

    Technologies_base=CONSTANTS["Technologies_base"]
    new_rows=[]
    Years_int=[2032,2024,2025,2026,2027,2028,2029,2031,2032,2033,2034,2035,2036,2037,2038,2039]
    for t in Technologies_base:
        for v in VECTO_Groups:
            Array_Eff_Years= Efficiencies[(Efficiencies["Technology"]==t) & (Efficiencies["VECTO_Group"]==v)]["Year"]
            Array_Eff_Eff= Efficiencies[(Efficiencies["Technology"]==t) & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"]
            for y in Years_int:
                Effi=np.interp(y,Array_Eff_Years,Array_Eff_Eff)
                temp_df_eff = {
                    "Technology": t, 
                    "VECTO_Group": v,
                    "Year": y,  
                    "Efficiency": 0,
                    "Efficiency_kWh": Effi
                }
                new_rows.append(temp_df_eff)
    new_df=pd.DataFrame(new_rows)
    Efficiencies = pd.concat([Efficiencies.reset_index(drop=True), new_df], axis=0)


    ng_rows = Efficiencies[Efficiencies['Technology'] == 'NG']
    bg_rows = ng_rows.copy()
    bg_rows['Technology'] = 'BG'
    FCET_rows = Efficiencies[Efficiencies['Technology'] == 'FCET']
    REFCET_rows = FCET_rows.copy()
    REFCET_rows['Technology'] = 'RE-FCET'
    dis_rows = Efficiencies[Efficiencies['Technology'] == 'Diesel']
    hvo_rows=dis_rows.copy()
    hvo_rows['Technology']= "HVO"
    Efficiencies = pd.concat([Efficiencies.reset_index(drop=True), REFCET_rows.reset_index(drop=True)], axis=0)
    Efficiencies = pd.concat([Efficiencies.reset_index(drop=True), bg_rows.reset_index(drop=True)], axis=0)
    Efficiencies= pd.concat([Efficiencies.reset_index(drop=True), hvo_rows.reset_index(drop=True)], axis=0)
    
    interpolated_df = add_and_interpolate_all(Components_Price)
    Components_Price=(interpolated_df)
    ETS_scenarios=pd.Series(ETS_Price["Scenario"]).drop_duplicates().tolist()

    # Interpolation function
    def interpolate_value(year, column):
        return np.interp(year, Electrolyzer_IEA["Year"], Electrolyzer_IEA[column])

    # CAPEX per kg H2 computation
    def compute_capex_per_kg(iea_capex, lifetime, efficiency):
        lifetime_years = lifetime / CONSTANTS["hours_year_electrolyzer"]
        npv_factor = (1 - (1 + CONSTANTS["DR"]) ** -lifetime_years) / CONSTANTS["DR"]
        return iea_capex / (npv_factor * (CONSTANTS["hours_year_electrolyzer"] / efficiency))
    
    def get_electricity_price(FuelPrices, year):
        prices=  FuelPrices[FuelPrices["Fuel"] == "Electricity"]
        prices = prices[prices["Year"] == year].set_index("Scenario")["Final_price"].to_dict()
        return prices  # Returns a dictionary with scenarios as keys
    # Hydrogen price computation
    def compute_hydrogen_price(year):
        electricity_prices = get_electricity_price(FuelPrices, year)
        efficiency = CONSTANTS["kWh_kg_H2"] / interpolate_value(year, "Efficiency")
        lifetime = interpolate_value(year, "Lifetime_hours")

        iea_capex_high = interpolate_value(year, "IEA_minimum")
        iea_capex_average = interpolate_value(year, "IEA_average")
        iea_capex_low = interpolate_value(year, "IEA_high")

        capex_scenarios = {
            "low": compute_capex_per_kg(iea_capex_low, lifetime, efficiency),
            "average": compute_capex_per_kg(iea_capex_average, lifetime, efficiency),
            "high": compute_capex_per_kg(iea_capex_high, lifetime, efficiency)
        }

        hydrogen_prices = []
        for capex_scenario, capex_value in capex_scenarios.items():
            for elec_scenario, elec_price in electricity_prices.items():
                opex = elec_price * efficiency
                hydrogen_prices.append({
                    "Year": year,
                    "CAPEX Scenario": capex_scenario,
                    "Electricity Scenario": elec_scenario,
                    "Electricity Price (€/kWh)": elec_price,
                    "CAPEX (€/kg)": capex_value,
                    "OPEX (€/kg)": opex,
                    "Total Hydrogen Price (€/kg)": capex_value + opex,
                    "Lifetime (years)": lifetime / CONSTANTS["hours_year_electrolyzer"],
                    "Efficiency": efficiency,
                    "IEA_Capex_Low": iea_capex_low,
                    "IEA_Capex_Average": iea_capex_average,
                    "IEA_Capex_High": iea_capex_high
                })

        return pd.DataFrame(hydrogen_prices)

    # Compute hydrogen prices for each year in Electrolyzer_IEA dataset
    hydrogen_price_data = pd.concat([compute_hydrogen_price(year) for year in Electrolyzer_IEA["Year"]], ignore_index=True)



    return {
        "Chassis_Price": Chassis_Price,
        "Components_Price": Components_Price,
        "Efficiencies": Efficiencies,
        "CAPEX_Vehicles": CAPEX_Vehicles,
        "Powertrain_Features": Powertrain_Features,
        "Maintenance_curve": Maintenance_curve,
        "Maintenance_Price": Maintenance_Price,
        "Em_Factors": Em_Factors,
        "ETS_Price": ETS_Price,
        "Comp_Features": Comp_Features,
        "Cost_Red_H2": Cost_Red_H2,
        "VECTO": VECTO,
        "FuelPrices": FuelPrices,  # Ensure this is loaded last,
        "Electrolyzer_IEA": Electrolyzer_IEA,
        "hydrogen_price_data": hydrogen_price_data
    }


def analyze_vehicle_data():
    base_path = "../data/"
    EEA = pd.read_csv(f"{base_path}EEA_Registration_Data.csv")
    
    EEA.loc[EEA["OEM_VehicleSubgroup"].isin(["Other", ""]), "OEM_VehicleSubgroup"] = EEA["OEM_VehicleGroup"]

    # Calculate MaxMass and PowerToWeightRatio
    EEA["MaxMass"] = (EEA["OEM_GrossVehicleMass_t"] * 1000 - EEA["CurbMassChassis_kg"])
    EEA["PowerToWeightRatio"] = (EEA["Engine_RatedPower_kw"] / EEA["MaxMass"])

    # Define order for vehicle subgroups
    order = ["1", "2", "3", "4-UD", "4-RD", "4-LH", "5-RD", "5-LH", "9-RD", "9-LH", "10-RD", "10-LH", "11", "12", "16"]

    # Remove rows with NaN values in relevant columns
    EEA = EEA.dropna(subset=["OEM_VehicleSubgroup", "Engine_RatedPower_kw"])

    # Create a boxplot of the power-to-weight ratio by OEM_VehicleSubgroup
    plt.figure(figsize=(10, 6))
    boxplot = sns.boxplot(x="OEM_VehicleSubgroup", y="Engine_RatedPower_kw", data=EEA, order=order)

    # Calculate and annotate the median values
    medians = EEA.groupby("OEM_VehicleSubgroup")["Engine_RatedPower_kw"].median()
    #print("Medians:", medians)

    plt.xlabel("OEM Vehicle VECTO Subgroup")
    plt.ylabel("Engine Rated Power (kW)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(results_folder, "Power_VECTO.svg"), format="svg")
    plt.show()

    # Calculate the percentage of entries for each OEM_VehicleSubgroup
    total_entries = len(EEA)
    subgroup_counts = EEA["OEM_VehicleSubgroup"].value_counts()
    subgroup_percentages = (subgroup_counts / total_entries) * 100

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=subgroup_percentages.index, y=subgroup_percentages.values, order=order, color="steelblue")
    plt.xlabel("OEM Vehicle VECTO Subgroup")
    plt.ylabel("Percentage of Registrations in 2022 (%)")

    plt.savefig(os.path.join(results_folder, "Distribution_VECTO_2024.svg"), format="svg")
    plt.show()

    # Select columns containing "km"
    distance_data = EEA.filter(like="km")
    EEA_Registrations=EEA
    EEA_VECTO_percentages= subgroup_percentages
    EEA_distance_data=distance_data
    return EEA_Registrations, EEA_Registrations, EEA_distance_data

def analyze_etis_data():
    base_path = "../data/"
    ETIS = pd.read_csv(f"{base_path}01_Trucktrafficflow.csv")
    ETIS_largos = ETIS[ETIS["Distance_within_E_road"] > 500]

    # Summarize traffic flow
    result = ETIS_largos.groupby("ID_origin_region").agg({"Traffic_flow_trucks_2030": "sum"}).reset_index()
    result2 = ETIS.groupby("ID_origin_region").agg({"Traffic_flow_trucks_2030": "sum"}).reset_index()

    # Merge results and calculate percentage
    result = result.merge(result2, on="ID_origin_region", suffixes=(".x", ".y"))
    result["porc"] = result["Traffic_flow_trucks_2030.x"] / result["Traffic_flow_trucks_2030.y"]

    # Summing Trucks for each Distance
    summarized_df = ETIS.groupby("Distance_within_E_road").agg({"Traffic_flow_trucks_2030": "sum"}).reset_index()

    # Data for the simpler histogram, taken from the Eurostat dataset
    simple_data = {
        "Category": ["<50 km", "50-149 km", "150-299 km", "300-499 km", "500-999 km", "1,000-1,999 km", "2,000-5,999 km"],
        "Count": [509480000, 237257000, 123910000, 58293000, 41542000, 12756000, 1934000],
    }
    hist_df = pd.DataFrame(simple_data)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # ETIS Histogram
    bin_width = 50
    bins = range(0, int(summarized_df["Distance_within_E_road"].max()) + bin_width, bin_width)
    sns.histplot(data=summarized_df, x="Distance_within_E_road", weights="Traffic_flow_trucks_2030",
                 bins=bins, kde=False, ax=axes[0], color="steelblue", edgecolor=None)
    
    axes[0].set_xlabel("Distance (km)", fontsize=12)
    axes[0].set_ylabel("Total operations", fontsize=12)
    axes[0].set_xlim(0, 1500)
    axes[0].set_title("ETIS Plus (2030)")
    axes[0].tick_params(axis="x", labelsize=12)
    axes[0].tick_params(axis="y", labelsize=12)

    # Eurostat Bar Chart
    sns.barplot(data=hist_df, x="Category", y="Count", ax=axes[1], color="darkorange")
    axes[1].set_xlabel("Distance", fontsize=12)
    axes[1].set_ylabel("Total operations", fontsize=12)
    axes[1].set_xticklabels(hist_df["Category"], rotation=45, ha="right", fontsize=12)
    axes[1].set_title("Eurostat (2023)")
    axes[1].tick_params(axis="x", labelsize=12)
    axes[1].tick_params(axis="y", labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_folder, "Distance_Distribution.svg"), format="svg")

    plt.show()
    ETIS_result_table= result
    ETIS_summarized_table= summarized_df
    return ETIS_result_table, ETIS_summarized_table

def plot_fuel_price_components(CONSTANTS, years_to_plot, data):
    palette = CONSTANTS["palette"]
    # Define the column names
    columns = ['Fuel', 'Scenario', 'Year', 'Energy', 'Energy_Taxes', 'Distribution', 'Final_price']
    df = pd.DataFrame(data["FuelPrices"], columns=columns)

    # Rename fuels
    fuel_rename = {
        'Oil': 'Diesel',
        'NG': 'CNG',
        'Electricity': 'Electricity',
        'BG': 'Biomethane',
        'HVO': 'HVO',
        'H2': 'H2'
    }
    df['Fuel'] = df['Fuel'].replace(fuel_rename)

    # Filter desired years and scenario
    df_filtered = df[(df['Year'].isin(years_to_plot)) & (df['Scenario'] == "Average")]

    # Define colors for each price component
    component_colors = {
        'Energy': palette[0],
        'Energy_Taxes': palette[2],
        'Distribution': palette[3]
    }

    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.15  # Thinner bars
    spacing = 2  # Increase spacing between year groups
    x_labels = []
    x_positions = []
    year_positions_map = {}

    for i, year in enumerate(years_to_plot):
        df_year = df_filtered[df_filtered['Year'] == year]
        fuels = df_year['Fuel'].unique()
        bottom_values = np.zeros(len(fuels))  # For stacking
        year_positions = [i * spacing + j * (bar_width + 0.1) for j in range(len(fuels))]
        x_positions.extend(year_positions)
        x_labels.extend(fuels)
        year_positions_map[year] = np.mean(year_positions)  # Central position for each year group

        for component in ['Energy', 'Energy_Taxes', 'Distribution']:
            values = df_year.set_index('Fuel')[component].reindex(fuels).values
            ax.bar(year_positions, values, bar_width, label=component if i == 0 else "",
                   color=component_colors[component], edgecolor='grey', bottom=bottom_values)
            bottom_values += values  # Stack next component

            # Add error bars for Energy component
            if component == 'Distribution':
                high_scenario_energy = df[(df['Year'] == year) & (df['Scenario'] == 'High')].set_index('Fuel')['Energy'].reindex(fuels).values
                low_scenario_energy = df[(df['Year'] == year) & (df['Scenario'] == 'Low')].set_index('Fuel')['Energy'].reindex(fuels).values
                error_bars_energy = [df_year.set_index('Fuel')['Energy'].reindex(fuels).values - low_scenario_energy, high_scenario_energy - df_year.set_index('Fuel')['Energy'].reindex(fuels).values]
                ax.errorbar(year_positions, bottom_values, yerr=error_bars_energy, fmt='none', ecolor='black', capsize=5)

    # Set plot parameters
    plt.rcParams.update({'font.size': 10})  # Set font size to 10
    ax.set_ylabel('EUR/kWh')
    ax.legend()
    ax.grid(axis='y', linestyle='', alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  # Increase upper limit of Y-axis

    # Add year labels below the fuel group
    for year, xpos in year_positions_map.items():
        ax.text(xpos, -ax.get_ylim()[1] * 0.2, str(year), ha='center', va='top', fontsize=16)

    # Save plot if required
    plt.savefig(os.path.join(results_folder, 'Fuel_Price_Components.svg'), format='svg')

    # Show the plot
    plt.show()

    return ax

