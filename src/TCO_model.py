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

def BET_CAPEX_components(Year, Case, Scenario,data,CONSTANTS):
    Efficiencies=data["Efficiencies"]
    Chassis_Price=data["Chassis_Price"]
    Components_Price=data["Components_Price"]
    VECTO=data["VECTO"]
    Powertrain_Features=data["Powertrain_Features"]
    Technology = "BET"
    VECTO_Group = Case[1]["VECTO_Group"]
    Efficiency=Efficiencies[(Efficiencies["Year"]==Year) & (Efficiencies["Technology"]=="BET") & (Efficiencies["VECTO_Group"]==VECTO_Group)]["Efficiency_kWh"].iat[0]
    Chassis = float(Chassis_Price[Chassis_Price["VECTO_Group"] == VECTO_Group]["Price_Eur"].iat[0])
    Power = float(VECTO[VECTO["VECTO_Group"] == VECTO_Group]["Engine_Power_kW"].iat[0])
    Battery_Capacity_kWh=Case[1]["Distance"]/Efficiency
    Battery_Price_eur_kWh = float(Components_Price[(Components_Price["Component"] == "Energy battery") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == Scenario)]["Price"].iat[0])
    Power_Elec_Price = float(Components_Price[(Components_Price["Component"] == "Power electronics") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == "ICCT_Central")]["Price"].iat[0])
    Elc_Drive_Price = float(Components_Price[(Components_Price["Component"] == "Electric drive") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == "ICCT_Central")]["Price"].iat[0])
    
    Chassis_cost = Chassis
    Battery_cost = Battery_Capacity_kWh * Battery_Price_eur_kWh
    Power_electronics_cost = Power * Power_Elec_Price
    Electric_drive_cost = Power * Elc_Drive_Price
    
    BEV_Price = CONSTANTS["Indirect_cost"] * (Chassis_cost + Battery_cost + Power_electronics_cost + Electric_drive_cost)
    
    return {
        "Year":Year,
        "Chassis": Chassis_cost,
        "Battery": Battery_cost,
        "Power Electronics": Power_electronics_cost,
        "Electric Drive": Electric_drive_cost,
        "Indirect costs": (CONSTANTS["Indirect_cost"]-1)*(Chassis_cost + Battery_cost + Power_electronics_cost + Electric_drive_cost),
        "Residual cost": -CONSTANTS["RValue"]*BEV_Price,
        "Total": BEV_Price
    }

def FCEV_CAPEX_components(Year, Case, Scenario, data, CONSTANTS):
    Efficiencies=data["Efficiencies"]
    Chassis_Price=data["Chassis_Price"]
    Components_Price=data["Components_Price"]
    VECTO=data["VECTO"]
    Powertrain_Features=data["Powertrain_Features"]
    VECTO_Group=Case[1]["VECTO_Group"]

    Efficiency=Efficiencies[(Efficiencies["Year"]==Year) & (Efficiencies["Technology"]=="FCET") & (Efficiencies["VECTO_Group"]==VECTO_Group)]["Efficiency_kWh"].iat[0]
    Chassis = float(Chassis_Price[Chassis_Price["VECTO_Group"] == VECTO_Group]["Price_Eur"].iat[0])
    Power = float(VECTO[VECTO["VECTO_Group"] == VECTO_Group]["Engine_Power_kW"].iat[0])
    FC_Power=0.55*Power+15
    #FC_Power = float(Powertrain_Features[(Powertrain_Features["VECTO_Group"] == VECTO_Group) & (Powertrain_Features["Technology"] == Technology) & (Powertrain_Features["Feature"] == "FC_Power")]["Capacity"].iat[0])
    FC_Price_eur_kW = float(Components_Price[(Components_Price["Component"] == "Fuel cell stack") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == Scenario)]["Price"].iat[0])
    FC_Price = FC_Power * FC_Price_eur_kW
    Battery_Capacity_kWh=100
    #Array_PT_Years = Powertrain_Features[(Powertrain_Features["Technology"] == Technology) & (Powertrain_Features["VECTO_Group"] == VECTO_Group) & (Powertrain_Features["Feature"] == "Battery_kWh")]["Year"]
    #Array_PT_Capacity = Powertrain_Features[(Powertrain_Features["Technology"] == Technology) & (Powertrain_Features["VECTO_Group"] == VECTO_Group) & (Powertrain_Features["Feature"] == "Battery_kWh")]["Capacity"]
    #Battery_Capacity_kWh = np.interp(Year, Array_PT_Years, Array_PT_Capacity)
    Battery_Price_eur_kWh = float(Components_Price[(Components_Price["Component"] == "Power Battery") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == Scenario)]["Price"].iat[0])
    Battery_price = Battery_Capacity_kWh * Battery_Price_eur_kWh

    H2_Capacity_kg= ((Case[1]["Distance"]*Efficiency)/CONSTANTS["kWh_kg_H2"])
    #print(Case[1]["Distance"])
    #print("Eff", Efficiency)
    #print("H2_Capacity_kg", H2_Capacity_kg)

    H2_capacity_price = float(Components_Price[(Components_Price["Component"] == "H2_storage") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == "ICCT_Central")]["Price"].iat[0])

    H2_tank_price = H2_Capacity_kg * H2_capacity_price
    Power_Elec_Price = float(Components_Price[(Components_Price["Component"] == "Power electronics") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == "ICCT_Central")]["Price"].iat[0])
    Elc_Drive_Price = float(Components_Price[(Components_Price["Component"] == "Electric drive") & (Components_Price["Year"] == Year) & (Components_Price["Scenario"] == "ICCT_Central")]["Price"].iat[0])
    PE_ED_price = Power * Power_Elec_Price + Power * Elc_Drive_Price

    Chassis_cost = Chassis
    Battery_cost = Battery_price
    Power_electronics_cost = Power * Power_Elec_Price
    Electric_drive_cost = Power * Elc_Drive_Price
    H2_tank_cost = H2_tank_price
    Fuel_cell_cost = FC_Price

    FCEV_Price = CONSTANTS["Indirect_cost"] * (Chassis_cost + Battery_cost + Power_electronics_cost + Electric_drive_cost + H2_tank_cost + Fuel_cell_cost)

    return {
        "Year":Year,
        "Chassis": Chassis_cost,
        "Battery": Battery_cost,
        "Power Electronics": Power_electronics_cost,
        "Electric Drive": Electric_drive_cost,
        "H2 Tank": H2_tank_cost,
        "Fuel Cell": Fuel_cell_cost,
        "Indirect costs": (CONSTANTS["Indirect_cost"]-1)*(Chassis_cost + Battery_cost + Power_electronics_cost + Electric_drive_cost + H2_tank_cost + Fuel_cell_cost),
        "Residual cost": -FCEV_Price*CONSTANTS["RValue"],
        "Total": FCEV_Price
    }

def calculate_CAPEX_components_for_all_years(Case, Scenario, years, data, CONSTANTS):
    BET_components = []
    FCEV_components = []
    for year in years:
        BET_components.append(BET_CAPEX_components(year, Case, Scenario, data, CONSTANTS))
        FCEV_components.append(FCEV_CAPEX_components(year, Case, Scenario, data, CONSTANTS))
    return BET_components, FCEV_components


def plot_CAPEX_composition_by_component(years, Case, Central_Scenario, data, CONSTANTS, ax):
    Efficiencies = data["Efficiencies"]
    Chassis_Price = data["Chassis_Price"]
    Components_Price = data["Components_Price"]
    VECTO = data["VECTO"]
    Powertrain_Features = data["Powertrain_Features"]
    CAPEX_Vehicles = data["CAPEX_Vehicles"]
    VECTO_Group = Case[1]["VECTO_Group"]
    BET_colors = CONSTANTS["palette"][0:4] + [CONSTANTS["palette"][6]]
    FCEV_colors = CONSTANTS["palette"][0:7]
    BET_hatch = '..'
    FCEV_hatch = '//'
    hatch_color = 'grey'

    scenarios = ['All', 'ICCT_Central', 'Scientific', 'Near market']

    BET_components_all_scenarios, FCEV_components_all_scenarios = calculate_CAPEX_components_for_all_years(Case, Central_Scenario, years, data, CONSTANTS)

    # Convert lists to DataFrames
    BET_components_all_scenarios = pd.DataFrame(BET_components_all_scenarios)
    FCEV_components_all_scenarios = pd.DataFrame(FCEV_components_all_scenarios)

    width = 0.35  # the width of the bars
    gap = 0.05    # the gap between the bars
    x = np.arange(len(years))  # the label locations

    # Plotting BET components
    bottom_BET = np.zeros(len(years))
    for i, column in enumerate(BET_components_all_scenarios.drop(columns=["Total", "Year", "Residual cost"]).columns):
        bars = ax.bar(x - (width + gap) / 2, BET_components_all_scenarios[column], width, label=f'BET {column}', bottom=bottom_BET, color=BET_colors[i])
        for bar in bars:
            bar.set_hatch(BET_hatch)
            bar.set_edgecolor(hatch_color)
        bottom_BET += BET_components_all_scenarios[column]

    # Plotting FCEV components
    bottom_FCEV = np.zeros(len(years))
    for i, column in enumerate(FCEV_components_all_scenarios.drop(columns=["Total", "Year", "Residual cost"]).columns):
        bars = ax.bar(x + (width + gap) / 2, FCEV_components_all_scenarios[column], width, label=f'FCEV {column}', bottom=bottom_FCEV, color=FCEV_colors[i])
        for bar in bars:
            bar.set_hatch(FCEV_hatch)
            bar.set_edgecolor(hatch_color)
        bottom_FCEV += FCEV_components_all_scenarios[column]

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x / 1000)}'))
    ax.set_xlabel('Year')
    ax.set_ylabel('CAPEX (Thousand EUR)')
    ax.set_title(Case[1]["Case"])
    ax.set_xticks(x)
    ax.set_xticklabels(years)

    # Adding benchmark lines
    diesel_benchmark = CAPEX_Vehicles[(CAPEX_Vehicles["Technology"] == "Diesel") & (CAPEX_Vehicles["VECTO_Group"] == Case[1]["VECTO_Group"]) & (CAPEX_Vehicles["Year"].isin(years))]["Value"].values
    ng_benchmark = CAPEX_Vehicles[(CAPEX_Vehicles["Technology"] == "NG") & (CAPEX_Vehicles["VECTO_Group"] == Case[1]["VECTO_Group"]) & (CAPEX_Vehicles["Year"].isin(years))]["Value"].values

    ax.axhline(y=diesel_benchmark, color='black', linestyle='--', label='Diesel Benchmark')
    ax.axhline(y=ng_benchmark, color='black', linestyle='-.', label='NG Benchmark')

    # Update legend to include benchmark lines
    BET_totals_per_year = []
    FCEV_totals_per_year = []

    for year in years:
        BET_totals = []
        FCEV_totals = []

        for scenario in scenarios:
            BET_total_cost = BET_CAPEX_components(year, Case, scenario, data, CONSTANTS)["Total"]
            FCEV_total_cost = FCEV_CAPEX_components(year, Case, scenario, data, CONSTANTS)["Total"]

            BET_totals.append(BET_total_cost)
            FCEV_totals.append(FCEV_total_cost)

        BET_totals_per_year.append(BET_totals)
        FCEV_totals_per_year.append(FCEV_totals)

    BET_min_totals_per_year = np.min(BET_totals_per_year, axis=1)
    BET_max_totals_per_year = np.max(BET_totals_per_year, axis=1)

    FCEV_min_totals_per_year = np.min(FCEV_totals_per_year, axis=1)
    FCEV_max_totals_per_year = np.max(FCEV_totals_per_year, axis=1)

    BET_errors_per_year = BET_max_totals_per_year - BET_min_totals_per_year
    FCEV_errors_per_year = FCEV_max_totals_per_year - FCEV_min_totals_per_year

    # Add error bars to the plot
    ax.errorbar(x - (width + gap) / 2, BET_max_totals_per_year, yerr=BET_errors_per_year, fmt='none', ecolor='black', capsize=5)
    ax.errorbar(x + (width + gap) / 2, FCEV_max_totals_per_year, yerr=FCEV_errors_per_year, fmt='none', ecolor='black', capsize=5)

def calculate_all_CAPEX(Cases, data, CONSTANTS):
    Life=CONSTANTS["Life"]
    Rate=CONSTANTS["Rate"]
    leverage=CONSTANTS["leverage"]
    DR=CONSTANTS["DR"]
    nper=CONSTANTS["nper"]
    Years=CONSTANTS["Years"]
    CAPEX_Vehicles=data["CAPEX_Vehicles"]

    Technologies_conv=["Diesel", "NG", "BG", "HVO"]
    New_Techs=["BET", "FCET", "RE-FCET"]
    new_rows=[]
    Years_int=[2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039, 2040]
    for c in Cases.iterrows():
        for t in Technologies_conv:
            Array_Cap_Years= CAPEX_Vehicles[(CAPEX_Vehicles["Technology"]==t) & (CAPEX_Vehicles["VECTO_Group"]==c[1]["VECTO_Group"])]["Year"]
            Array_Cap_Eff= CAPEX_Vehicles[(CAPEX_Vehicles["Technology"]==t) & (CAPEX_Vehicles["VECTO_Group"]==c[1]["VECTO_Group"])]["Value"]
            for y in Years_int:
                Capex_temp=np.interp(y,Array_Cap_Years,Array_Cap_Eff)
                ResidValue=CONSTANTS["RValue"]*Capex_temp
                pago=npf.pmt(Rate,nper,Capex_temp*(1-leverage))
                pagos=-np.ones(nper)*pago
                pagos[Life-1]=pagos[Life-1]+ResidValue
                pagos_npv=npf.npv(DR,pagos)
                Capex_npv=pagos_npv+(Capex_temp*leverage)
                temp_df_cap = {
                    "Case": c[1]["Case"],
                    "Technology": t, 
                    "VECTO_Group": c[1]["VECTO_Group"],
                    "Year": y,
                    "Scenario":"Average",  
                    "Component":"CAPEX",
                    "Value": Capex_npv
                }
                temp_df_pprice = {
                    "Case": c[1]["Case"],
                    "Technology": t, 
                    "VECTO_Group": c[1]["VECTO_Group"],
                    "Year": y,
                    "Scenario":"Average",  
                    "Component":"Purchase_Price",
                    "Value": Capex_temp
                }
                new_rows.append(temp_df_cap)
                new_rows.append(temp_df_pprice)
    new_df=pd.DataFrame(new_rows)
    All_CAPEX = new_df

    new_rows=[]
    Scenarios=["Near market", "All", "Scientific", "ICCT_Central"]
    for c in Cases.iterrows():
        for y in Years:
            for s in Scenarios:
                v=c[1]["VECTO_Group"]
                #print(c)
                Capex_BET=BET_CAPEX_components(y,c,s, data, CONSTANTS)["Total"]
                pago=npf.pmt(Rate,nper,Capex_BET*(1-leverage))
                pagos=-np.ones(nper)*pago
                ResidVal=BET_CAPEX_components(y,c,s, data, CONSTANTS)["Residual cost"]
                pagos[Life-1]=pagos[Life-1]+ResidVal
                pagos_npv=npf.npv(DR,pagos)
                Capex_npv_BET=pagos_npv+(Capex_BET*leverage)
                Capex_FCET=FCEV_CAPEX_components(y,c,s, data, CONSTANTS)["Total"]
                pago=npf.pmt(Rate,nper,Capex_FCET*(1-leverage))
                ResidVal=FCEV_CAPEX_components(y,c,s, data, CONSTANTS)["Residual cost"]
                pagos[Life-1]=pagos[Life-1]+ResidVal
                pagos=-np.ones(nper)*pago
                pagos_npv=npf.npv(DR,pagos)
                Capex_npv_FCET=pagos_npv+(Capex_FCET*leverage)
                Capex_REFCET=0.7*FCEV_CAPEX_components(y,c,s, data, CONSTANTS)["Total"]
                pago=npf.pmt(Rate,nper,Capex_REFCET*(1-leverage))
                pagos=-np.ones(nper)*pago
                ResidVal=0.7*FCEV_CAPEX_components(y,c,s, data, CONSTANTS)["Residual cost"]
                pagos[Life-1]=pagos[Life-1]+ResidVal
                pagos_npv=npf.npv(DR,pagos)
                Capex_npv_REFCET=pagos_npv+(Capex_REFCET*leverage)
                temp_df_cap_BET = {
                    "Case": c[1]["Case"],
                    "Technology": "BET", 
                    "VECTO_Group": v,
                    "Year": y,
                    "Scenario":s,  
                    "Component":"CAPEX",
                    "Value": Capex_npv_BET
                }
                temp_df_cap_FCET = {
                    "Case": c[1]["Case"],
                    "Technology": "FCET", 
                    "VECTO_Group": v,
                    "Year": y,
                    "Scenario":s,  
                    "Component":"CAPEX",
                    "Value": Capex_npv_FCET
                }
                temp_df_cap_REFCET = {
                    "Case": c[1]["Case"],
                    "Technology": "RE-FCET", 
                    "VECTO_Group": v,
                    "Year": y,
                    "Scenario":s,  
                    "Component":"CAPEX",
                    "Value": Capex_npv_REFCET
                }
                temp_df_pprice_BET = {
                    "Case": c[1]["Case"],
                    "Technology": "BET", 
                    "VECTO_Group": v,
                    "Year": y,
                    "Scenario":s,  
                    "Component":"Purchase_Price",
                    "Value": Capex_BET
                }
                temp_df_pprice_FCET = {
                    "Case": c[1]["Case"],
                    "Technology": "FCET", 
                    "VECTO_Group": v,
                    "Year": y,
                    "Scenario":s,  
                    "Component":"Purchase_Price",
                    "Value": Capex_FCET
                }
                temp_df_pprice_REFCET = {
                    "Case": c[1]["Case"],
                    "Technology": "RE-FCET", 
                    "VECTO_Group": v,
                    "Year": y,
                    "Scenario":s,  
                    "Component":"Purchase_Price",
                    "Value": Capex_REFCET
                }
                new_rows.append(temp_df_cap_BET)
                new_rows.append(temp_df_cap_FCET)
                new_rows.append(temp_df_cap_REFCET) 
                new_rows.append(temp_df_pprice_BET)
                new_rows.append(temp_df_pprice_FCET)
                new_rows.append(temp_df_pprice_REFCET)    
    new_df=pd.DataFrame(new_rows)
    All_CAPEX = pd.concat([All_CAPEX.reset_index(drop=True), new_df], axis=0)
    CAPEX_df=All_CAPEX
    return CAPEX_df

def calculate_FIXOM(Cases, CAPEX_df, data, CONSTANTS):
    Technologies=["Diesel", "NG", "BG", "BET", "FCET", "RE-FCET", "HVO"]
    Years=np.arange(2024,2041,1, dtype=int)
    Life=CONSTANTS["Life"]
    FIXOM=[]
    CAPEX_scenarios = pd.Series(CAPEX_df["Scenario"]).drop_duplicates().tolist()
    for c in Cases.iterrows():
        for t in Technologies:
            for y in Years:
                for sca in CAPEX_scenarios:
                    # Filter the DataFrame for the current combination
                    filtered_df = CAPEX_df[(CAPEX_df["Technology"] == t) & 
                                                (CAPEX_df["VECTO_Group"] == c[1]["VECTO_Group"])& 
                                                (CAPEX_df["Year"] == y) & 
                                                (CAPEX_df["Scenario"] == sca)]
                    # Check if the filtered DataFrame is not empty
                    if not filtered_df.empty:
                        Insurance = CONSTANTS["Insurance_rate"] * np.ones(Life) * filtered_df["Value"].iat[0]
                        Insurance_npv = npf.npv(CONSTANTS["DR"], Insurance)
                        fixom = Insurance_npv
                        temp_df_fixom = {
                            "Case": c[1]["Case"],
                            "Technology": t, 
                            "VECTO_Group": c[1]["VECTO_Group"],
                            "Year": y, 
                            "Scenario": sca, 
                            "Component": "FIXOM", 
                            "Value": fixom
                        }
                        FIXOM.append(temp_df_fixom)
                        #print(len(FIXOM))
    FIXOM_df = pd.DataFrame(FIXOM)
    
    return FIXOM_df

def calculate_VAROM(Cases, CAPEX_df, data, CONSTANTS):
    Maintenance_Price=data["Maintenance_Price"]
    Maintenance_curve=data["Maintenance_curve"]
    VECTO=data["VECTO"]
    Technologies=["Diesel", "NG", "BG", "BET", "FCET", "RE-FCET", "HVO"]
    Years=np.arange(2024,2041,1, dtype=int)
    Life=CONSTANTS["Life"]
    VAROM=[]
    for c in Cases.iterrows():
        for t in Technologies:
            v=c[1]["VECTO_Group"]
            for y in Years:
                # Filter the DataFrame for the current combination
                maintenance_price_df = Maintenance_Price[(Maintenance_Price["Technology"] == t) & 
                                                        (Maintenance_Price["VECTO_Group"] == v)]
                vecto_df = VECTO[VECTO["VECTO_Group"] == v]
                
                # Check if the filtered DataFrames are not empty
                if not maintenance_price_df.empty and not vecto_df.empty:
                    cost_km = maintenance_price_df["Price"].iat[0] / 100
                    cost_maintenance =  np.ones(Life) *cost_km * c[1]["Distance"]*CONSTANTS["working_days"] * Maintenance_curve["Coeff"][1:Life+1]
                    maintenance_npv = npf.npv(CONSTANTS["DR"], cost_maintenance)
                    
                    tolls = np.ones(Life) * c[1]["Distance"]*CONSTANTS["working_days"]* (CONSTANTS["Road_tolls"] / 100)
                    tolls_npv = npf.npv(CONSTANTS["DR"], tolls)
                    
                    varom = maintenance_npv + tolls_npv
                    temp_df_varom = {
                        "Case": c[1]["Case"],
                        "Technology": t, 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Average", 
                        "Component": "VAROM", 
                        "Value": varom
                    }
                    VAROM.append(temp_df_varom)
                    if t=="NG":
                        temp_df_varom = {
                            "Case": c[1]["Case"],
                            "Technology": "BG", 
                            "VECTO_Group": v,
                            "Year": y, 
                            "Scenario": "Average", 
                            "Component": "VAROM", 
                            "Value": varom
                        }
                        VAROM.append(temp_df_varom)
                    if t=="FCET":
                        temp_df_varom = {
                            "Case": c[1]["Case"],
                            "Technology": "RE-FCET", 
                            "VECTO_Group": v,
                            "Year": y, 
                            "Scenario": "Average", 
                            "Component": "VAROM", 
                            "Value": varom
                        }
                        VAROM.append(temp_df_varom)
                    if t=="Diesel":
                        temp_df_varom = {
                            "Case": c[1]["Case"],
                            "Technology": "HVO", 
                            "VECTO_Group": v,
                            "Year": y, 
                            "Scenario": "Average", 
                            "Component": "VAROM", 
                            "Value": varom
                        }
                        VAROM.append(temp_df_varom)

    VAROM_df = pd.DataFrame(VAROM)
    
    return VAROM_df

def calculate_energy(Cases, data, CONSTANTS):
    FuelPrices=data["FuelPrices"]
    Efficiencies=data["Efficiencies"]
    Life=CONSTANTS["Life"]
    DR=CONSTANTS["DR"]
    working_days=CONSTANTS["working_days"]
    Years=CONSTANTS["Years"]

    results=[]
    Electricity_scenarios=pd.Series(FuelPrices["Scenario"]).drop_duplicates().tolist()
    Fossil_scenarios=pd.Series(FuelPrices["Scenario"]).drop_duplicates().tolist()
    H2_scenarios=pd.Series(FuelPrices["Scenario"]).drop_duplicates().tolist()

    for c in Cases.iterrows():
        v=c[1]["VECTO_Group"]
        for y in Years:
            Eff_Diesel=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="Diesel") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_NG=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="NG") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_BG=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="BG") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_BET=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="BET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_FCET=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="FCET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_REFCET=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="RE-FCET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]

            years = np.arange(y, y + Life)
            for esc in Electricity_scenarios:
                elc = FuelPrices[(FuelPrices["Year"].isin(years)) & 
                                    (FuelPrices["Scenario"] == esc) & (FuelPrices["Fuel"]=="Electricity")]["Final_price"]
                if c[1]["Case"]=="Long Haul - Long Distance":
                    cost_energy_elec=elc*c[1]["Distance"]*working_days*Eff_BET
                else:
                    cost_energy_elec=elc*c[1]["Distance"]*working_days*Eff_BET
                if not elc.empty:
                    npv_value = npf.npv(DR, cost_energy_elec)
                else:
                    npv_value = np.nan  # Or handle this case differently if needed

                temp_df_elc = {
                    "Case": c[1]["Case"],
                    "Technology": "BET", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": esc, 
                    "Component": "Energy", 
                    "Value": npv_value
                }
                results.append(temp_df_elc)
            for fsc in Fossil_scenarios:
                dis = FuelPrices[(FuelPrices["Year"].isin(years)) & 
                                    (FuelPrices["Scenario"] == fsc) & 
                                    (FuelPrices["Fuel"] == "Oil")]["Final_price"]
                cost_energy_diesel=dis*c[1]["Distance"]*working_days*Eff_Diesel
                hvo = FuelPrices[(FuelPrices["Year"].isin(years)) & 
                                    (FuelPrices["Scenario"] == fsc) & 
                                    (FuelPrices["Fuel"] == "HVO")]["Final_price"]
                cost_energy_hvo=hvo*c[1]["Distance"]*working_days*Eff_Diesel
                if not dis.empty:
                    npv_value = npf.npv(DR, cost_energy_diesel)
                    npv_value_hvo=npf.npv(DR, cost_energy_hvo)
                else:
                    npv_value = np.nan  # Or handle this case differently if needed

                temp_df_dis = {
                    "Case": c[1]["Case"],
                    "Technology": "Diesel", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": fsc, 
                    "Component": "Energy", 
                    "Value": npv_value
                }
                temp_df_HVO = {
                    "Case": c[1]["Case"],
                    "Technology": "HVO", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": fsc, 
                    "Component": "Energy", 
                    "Value": npv_value_hvo
                }
                gas = FuelPrices[(FuelPrices["Year"].isin(years)) & 
                                    (FuelPrices["Scenario"] == fsc) & 
                                    (FuelPrices["Fuel"] == "NG")]["Final_price"]
                cost_energy_gas=gas*c[1]["Distance"]*working_days*Eff_NG
                bgas = FuelPrices[(FuelPrices["Year"].isin(years)) & 
                                    (FuelPrices["Scenario"] == fsc) & 
                                    (FuelPrices["Fuel"] == "BG")]["Final_price"]
                cost_energy_bgas=bgas*c[1]["Distance"]*working_days*Eff_NG
                if not gas.empty:
                    npv_value = npf.npv(DR, cost_energy_gas)
                    npv_value_bgas=npf.npv(DR, cost_energy_bgas)
                else:
                    npv_value = np.nan  # Or handle this case differently if needed

                temp_df_gas = {
                    "Case": c[1]["Case"],
                    "Technology": "NG", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": fsc, 
                    "Component": "Energy", 
                    "Value": npv_value
                }
                temp_df_bgas = {
                    "Case": c[1]["Case"],
                    "Technology": "BG", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": fsc, 
                    "Component": "Energy", 
                    "Value": npv_value_bgas
                }                       
                results.append(temp_df_gas)
                results.append(temp_df_dis)
                results.append(temp_df_bgas)
                results.append(temp_df_HVO)

            for eh2 in H2_scenarios:
                h2 = FuelPrices[(FuelPrices["Year"].isin(years)) & 
                                    (FuelPrices["Scenario"] == eh2) & (FuelPrices["Fuel"]=="H2")]["Final_price"]
                cost_energy_h2=h2*c[1]["Distance"]*working_days*Eff_FCET

                if not h2.empty:
                    npv_value = npf.npv(DR, cost_energy_h2)
                else:
                    npv_value = np.nan  # Or handle this case differently if needed

                temp_df_h2 = {
                    "Case": c[1]["Case"],
                    "Technology": "FCET", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": eh2, 
                    "Component": "Energy", 
                    "Value": npv_value
                }
                temp_df_reh2 = {
                    "Case": c[1]["Case"],
                    "Technology": "RE-FCET", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": eh2, 
                    "Component": "Energy", 
                    "Value": npv_value
                }
                results.append(temp_df_h2)
                results.append(temp_df_reh2)    
    Energy_df= pd.DataFrame(results)
    return Energy_df


def calculate_ETS(Cases, data, CONSTANTS):
    Technologies=["Diesel", "NG", "BG", "BET", "FCET", "RE-FCET", "HVO"]
    Technologies_ETS= ["Diesel", "NG"]
    results=[]
    FuelPrices=data["FuelPrices"]
    Efficiencies=data["Efficiencies"]
    Life=CONSTANTS["Life"]
    DR=CONSTANTS["DR"]
    working_days=CONSTANTS["working_days"]
    Years=CONSTANTS["Years"]
    ETS_scenarios=['Low', 'Average', 'High']
    ETS_Price=data["ETS_Price"]
    ETS_year=CONSTANTS["ETS_year"]
    Em_Factors=data["Em_Factors"]
    for c in Cases.iterrows():
        for y in Years:
            v=c[1]["VECTO_Group"]
            Eff_Diesel=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="Diesel") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_NG=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="NG") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_BG=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="BG") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_BET=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="BET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_FCET=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="FCET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Eff_REFCET=Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="RE-FCET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]

            years = np.arange(y, y + Life)
            for etssc in ETS_scenarios:
                for t in Technologies:
                    if (t in (Technologies_ETS)) and (y>ETS_year-1):
                        efficiency=Efficiencies[(Efficiencies["Technology"]==t) & (Efficiencies["Year"]==y) & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
                        ets_pr = ETS_Price[(ETS_Price["Year"].isin(years)) & 
                                            (ETS_Price["Scenario"] == etssc)]["Price_eur_tco2"]
                        cost_ets=(ets_pr/1000)*c[1]["Distance"]*working_days*efficiency*Em_Factors[Em_Factors["Technology"]==t]["FE"].iat[0]
                        #print(cost_ets)
                        if not ets_pr.empty:
                            npv_value = npf.npv(DR, cost_ets)
                        else:
                            npv_value = 0  
                            
                        temp_df_elc = {
                            "Case": c[1]["Case"],
                            "Technology": t, 
                            "VECTO_Group": v,
                            "Year": y, 
                            "Scenario": etssc, 
                            "Component": "ETS", 
                            "Value": npv_value
                        }
                        results.append(temp_df_elc)
                    else:
                        temp_df_elc = {
                            "Case": c[1]["Case"],
                            "Technology": t, 
                            "VECTO_Group": v,
                            "Year": y, 
                            "Scenario": etssc, 
                            "Component": "ETS", 
                            "Value": 0
                        }
                        results.append(temp_df_elc)
    ETS_df = pd.DataFrame(results)
    return ETS_df

def Calculate_HRS_Cost(y,v,Vehicles_day, data, CONSTANTS):
    VECTO=data["VECTO"]
    Efficiencies=data["Efficiencies"]
    Powertrain_Features=data["Powertrain_Features"]
    Comp_Features=data["Comp_Features"]
    Cost_Red_H2=data["Cost_Red_H2"]
    kWh_kg_H2=CONSTANTS["kWh_kg_H2"]
    FuelPrices=data["FuelPrices"]
    Cap_truck_H2=300 #kg
    Hours_Refuelling=14
    Security_Factor=1.2
    Working_days=300
    Consumption_year_vehicle=VECTO[VECTO["VECTO_Group"]==v]["km_year"].iat[0]*Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="FCET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
    H2_day_demand=(Vehicles_day*(Consumption_year_vehicle/Working_days))
    H2_day_demand_dimensioning=H2_day_demand*(1/0.6)
    Dolar_Euro=1.09
    Dispenser=1
    HRS_Life=20
    HRS_DR=0.08
    N_trucks= 1

    High_Storage_Pressure=VECTO[VECTO["VECTO_Group"]==v]["Pressure"].iat[0]+200
    Refuelling_minutes_day=VECTO[VECTO["VECTO_Group"]==v]["Minutes_Refuelling"].iat[0]*Vehicles_day
    Low_storage_cap= H2_day_demand_dimensioning/kWh_kg_H2
    #print("LOWSTORAGE")
    #print(Low_storage_cap)
    Array_Tank_Years= Powertrain_Features[(Powertrain_Features["Technology"]=="FCEV") & (Powertrain_Features["VECTO_Group"]==v) & (Powertrain_Features["Feature"]=="H2_Tank")]["Year"]
    Array_Tank_Cap= Powertrain_Features[(Powertrain_Features["Technology"]=="FCEV") & (Powertrain_Features["VECTO_Group"]==v) & (Powertrain_Features["Feature"]=="H2_Tank")]["Capacity"]
    Tank_Capacity_kg=np.interp(y,Array_Tank_Years,Array_Tank_Cap)
    Refuelling_units_peak=np.ceil(Vehicles_day/Hours_Refuelling)
    #(Hours_Refuelling*60)/(VECTO[VECTO["VECTO_Group"]==v]["Minutes_Refuelling"].iat[0]+2)
    Peak_demand_kg=Refuelling_units_peak*Tank_Capacity_kg
    High_storage_cap=Peak_demand_kg*Security_Factor
    Compressor_Flow=((H2_day_demand_dimensioning/Hours_Refuelling)/3600)/kWh_kg_H2

    P_Compressor_CAPEX_kW=(8.314*(float(Comp_Features[Comp_Features["Feature"]=="Tin"]["Value"].iat[0])+273))/((2*0.4)*float(Comp_Features[Comp_Features["Feature"]=="Compressor_eff"]["Value"].iat[0]))*((((High_Storage_Pressure/((float(Comp_Features[Comp_Features["Feature"]=="Minimum_bar"]["Value"].iat[0]))))**(0.4/1.4))-1))*Compressor_Flow
    P_Compressor_OPEX_kW=(8.314*(float(Comp_Features[Comp_Features["Feature"]=="Tin"]["Value"].iat[0])+273))/((2*0.4)*float(Comp_Features[Comp_Features["Feature"]=="Compressor_eff"]["Value"].iat[0]))*((((High_Storage_Pressure/((float(Comp_Features[Comp_Features["Feature"]=="Low_Pressure_Storage"]["Value"].iat[0])-float(Comp_Features[Comp_Features["Feature"]=="Minimum_bar"]["Value"].iat[0]))/2))**(0.4/1.4))-1))*Compressor_Flow

    Array_Comp_CR_Years= Cost_Red_H2[(Cost_Red_H2["Component"]=="Comp_cost")]["Year"]
    Array_Comp_CR_Cost= Cost_Red_H2[(Cost_Red_H2["Component"]=="Comp_cost")]["Cost"]
    Comp_CR=np.interp(y,Array_Comp_CR_Years,Array_Comp_CR_Cost)
    CAPEX_Comp=40035*P_Compressor_CAPEX_kW**(0.6038)*Comp_CR

    Array_HS_CR_Years= Cost_Red_H2[(Cost_Red_H2["Component"]=="High_st_cost")]["Year"]
    Array_HS_CR_Cost= Cost_Red_H2[(Cost_Red_H2["Component"]=="High_st_cost")]["Cost"]
    HS_CR=np.interp(y,Array_HS_CR_Years,Array_HS_CR_Cost)
    CAPEX_HS=1800*Dolar_Euro*High_storage_cap*HS_CR

    Array_LS_CR_Years= Cost_Red_H2[(Cost_Red_H2["Component"]=="Low_st_cost")]["Year"]
    Array_LS_CR_Cost= Cost_Red_H2[(Cost_Red_H2["Component"]=="Low_st_cost")]["Cost"]
    LS_CR=np.interp(y,Array_LS_CR_Years,Array_LS_CR_Cost)
    CAPEX_LS=1100*Dolar_Euro*Low_storage_cap*LS_CR

    Array_Co_CR_Years= Cost_Red_H2[(Cost_Red_H2["Component"]=="Precooling_Unit")]["Year"]
    Array_Co_CR_Cost= Cost_Red_H2[(Cost_Red_H2["Component"]=="Precooling_Unit")]["Cost"]
    Co_CR=np.interp(y,Array_Co_CR_Years,Array_Co_CR_Cost)
    CAPEX_Co=140000*Dolar_Euro*Dispenser*Co_CR

    Array_Disp_CR_Years= Cost_Red_H2[(Cost_Red_H2["Component"]=="Disp_cost")]["Year"]
    Array_Disp_CR_Cost= Cost_Red_H2[(Cost_Red_H2["Component"]=="Disp_cost")]["Cost"]
    Disp_CR=np.interp(y,Array_Disp_CR_Years,Array_Disp_CR_Cost)
    CAPEX_Disp=100000*Dolar_Euro*Dispenser*Disp_CR

    Total_CAPEX=CAPEX_Comp+CAPEX_HS+CAPEX_LS+CAPEX_Co+CAPEX_Disp
    Annual_CAPEX=-npf.pmt(HRS_DR,HRS_Life,Total_CAPEX)
    Anuual_OPEX_Comp=P_Compressor_OPEX_kW*Working_days*Hours_Refuelling*FuelPrices[(FuelPrices["Scenario"]=="Average") & (FuelPrices["Year"]==y) & (FuelPrices["Fuel"]=="Electricity")]["Final_price"].iat[0]
    Annual_OPEX_Cool=(H2_day_demand_dimensioning/kWh_kg_H2)*Working_days*Comp_Features[Comp_Features["Feature"]=="Cooling_consump"]["Value"].iat[0]*FuelPrices[(FuelPrices["Scenario"]=="Average") & (FuelPrices["Year"]==y) & (FuelPrices["Fuel"]=="Electricity")]["Final_price"].iat[0]
    OandMCosts=0.05*Total_CAPEX

    HRS_CAPEX=(Annual_CAPEX/(Working_days*(H2_day_demand/kWh_kg_H2)))
    HRS_OPEX=(Anuual_OPEX_Comp+Annual_OPEX_Cool+OandMCosts)/(Working_days*(H2_day_demand/kWh_kg_H2))
    HRS_Cost=HRS_CAPEX+HRS_OPEX
    return HRS_Cost

def calculate_HRS(Cases, data, CONSTANTS):
    Years=CONSTANTS["Years"]
    Life=CONSTANTS["Life"]
    DR=CONSTANTS["DR"]
    working_days=CONSTANTS["working_days"]
    kWh_kg_H2=CONSTANTS["kWh_kg_H2"]
    Efficiencies=data["Efficiencies"]
    average_n=25
    HRS_results=[]
    n_list=np.arange(1,100,1)
    HRS_results=[]
    H2_Transport_Cost=1.6 #€/kg

    for c in Cases.iterrows():
        v=c[1]["VECTO_Group"]
        for y in Years:
            Annual_kWh=c[1]["Distance"]*working_days*Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="FCET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iloc[0]
            array_HRS = [Calculate_HRS_Cost(y,v,n, data, CONSTANTS) for n in n_list]
            min_HRS=min(array_HRS)+H2_Transport_Cost
            min_HRS_years=np.ones(Life)*min_HRS*(Annual_kWh/kWh_kg_H2)
            min_HRS_npv=npf.npv(DR,min_HRS_years)
            max_HRS=max(array_HRS)+H2_Transport_Cost
            max_HRS_years=np.ones(Life)*max_HRS*(Annual_kWh/kWh_kg_H2)
            max_HRS_npv=npf.npv(DR,max_HRS_years)
            av_HRS=Calculate_HRS_Cost(y,v,average_n, data, CONSTANTS)+H2_Transport_Cost
            av_HRS_years=np.ones(Life)*av_HRS
            av_HRS_npv=npf.npv(DR,av_HRS_years)*(Annual_kWh/kWh_kg_H2)
            min_HRS
            temp_df_max = {
                        "Case": c[1]["Case"],
                        "Technology": "FCET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "High", 
                        "Component": "Re-Infr", 
                        "Value": max_HRS_npv
                    }
            HRS_results.append(temp_df_max)
            temp_df_min = {
                        "Case": c[1]["Case"],
                        "Technology": "FCET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Low", 
                        "Component": "Re-Infr", 
                        "Value": min_HRS_npv
                    }
            HRS_results.append(temp_df_min)
            temp_df_av = {
                        "Case": c[1]["Case"],
                        "Technology": "FCET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Average", 
                        "Component": "Re-Infr", 
                        "Value": av_HRS_npv
                    }
            HRS_results.append(temp_df_av)
            temp_df_max = {
                        "Case": c[1]["Case"],
                        "Technology": "RE-FCET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "High", 
                        "Component": "Re-Infr", 
                        "Value": max_HRS_npv
                    }
            HRS_results.append(temp_df_max)
            temp_df_min = {
                        "Case": c[1]["Case"],
                        "Technology": "RE-FCET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Low", 
                        "Component": "Re-Infr", 
                        "Value": min_HRS_npv
                    }
            HRS_results.append(temp_df_min)
            temp_df_av = {
                        "Case": c[1]["Case"],
                        "Technology": "RE-FCET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Average", 
                        "Component": "Re-Infr", 
                        "Value": av_HRS_npv
                    }
            HRS_results.append(temp_df_av)
            for k in range(1, 100,1):
                concrete_HRS=Calculate_HRS_Cost(y,v,k, data, CONSTANTS)+H2_Transport_Cost
                concrete_HRS_years=np.ones(Life)*concrete_HRS
                concrete_HRS_npv=npf.npv(DR,concrete_HRS_years)*(Annual_kWh/kWh_kg_H2)
                temp_df_concrete = {
                    "Case": c[1]["Case"],
                    "Technology": "FCET", 
                    "VECTO_Group": v,
                    "Year": y, 
                    "Scenario": "F_{}".format(k), 
                    "Component": "Re-Infr", 
                    "Value": concrete_HRS_npv
                }
                HRS_results.append(temp_df_concrete)
    return HRS_results

def calculate_BET_charging(Cases, data, CONSTANTS):
    Years=CONSTANTS["Years"]
    Life=CONSTANTS["Life"]
    DR=CONSTANTS["DR"]
    working_days=CONSTANTS["working_days"]
    kWh_kg_H2=CONSTANTS["kWh_kg_H2"]
    Efficiencies=data["Efficiencies"]
    BET_CH_Results=[]
    Depot_price=CONSTANTS["Depot_price"]
    MW_Price=CONSTANTS["MW_Price"]


    for c in Cases.iterrows():
        v=c[1]["VECTO_Group"]
        for y in Years:
            Annual_kWh=c[1]["Distance"]*working_days*Efficiencies[(Efficiencies["Year"]==y) & (Efficiencies["Technology"]=="BET") & (Efficiencies["VECTO_Group"]==v)]["Efficiency_kWh"].iat[0]
            Annual_Depot=Annual_kWh*(Depot_price/1000)
            Annual_MW=Annual_kWh*(MW_Price/1000)
            Annual_MW_years=Annual_kWh*(MW_Price/1000)*np.ones(Life)
            Annual_Depot_years=Annual_kWh*(Depot_price/1000)*np.ones(Life)
            Annual_Average=Annual_Depot*c[1]["Depot"]+Annual_MW*c[1]["Public"]
            Annual_average_years=(Annual_Depot*c[1]["Depot"]+Annual_MW*c[1]["Public"])*np.ones(Life)
            npv_Depot=npf.npv(DR,Annual_Depot_years)    
            npv_MW=npf.npv(DR,Annual_MW_years)  
            npv_average=npf.npv(DR,Annual_average_years)      
            temp_df_min = {
                        "Case": c[1]["Case"],
                        "Technology": "BET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Low", 
                        "Component": "Re-Infr", 
                        "Value": npv_Depot
                    }
            BET_CH_Results.append(temp_df_min)
            temp_df_max = {
                        "Case": c[1]["Case"],
                        "Technology": "BET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "High", 
                        "Component": "Re-Infr", 
                        "Value": npv_MW
                    }
            BET_CH_Results.append(temp_df_max)
            temp_df_av = {
                        "Case": c[1]["Case"],
                        "Technology": "BET", 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Average", 
                        "Component": "Re-Infr", 
                        "Value": npv_average
                    }
            BET_CH_Results.append(temp_df_av)

    BET_CH_df=pd.DataFrame(BET_CH_Results)

    FF_CH_Results=[]

    for c in Cases.iterrows():
        v=c[1]["VECTO_Group"]
        for y in Years:
            for t in ["Diesel", "NG", "BG", "HVO"]:
                temp_df_FF = {
                        "Case": c[1]["Case"],
                        "Technology": t, 
                        "VECTO_Group": v,
                        "Year": y, 
                        "Scenario": "Average", 
                        "Component": "Re-Infr", 
                        "Value": 0
                    }
                FF_CH_Results.append(temp_df_FF)
    FF_CH_df=pd.DataFrame(FF_CH_Results)
    return BET_CH_df, FF_CH_df

def fill_missing_scenarios(df):
    filled_df = pd.DataFrame()
    for (tech, vecto, year, comp, case), group in df.groupby(['Technology', 'VECTO_Group', 'Year', 'Component', 'Case']):

        scenarios = group['Scenario'].unique()
        max_value = group['Value'].max()
        min_value = group['Value'].min()
        if 'High' not in scenarios:
            group = pd.concat([group, pd.DataFrame([{'Technology': tech, 'VECTO_Group': vecto, 'Year': year, 'Scenario': 'High', 'Component': comp, 'Value': max_value, 'Case': case}])])
        if 'Low' not in scenarios:
            group = pd.concat([group, pd.DataFrame([{'Technology': tech, 'VECTO_Group': vecto, 'Year': year, 'Scenario': 'Low', 'Component': comp, 'Value': min_value, 'Case':case}])])
        filled_df = pd.concat([filled_df, group])
    return filled_df

def make_Total_matrix(Cases, data, CONSTANTS):
    # Calculate all components
    CAPEX_df = calculate_all_CAPEX(Cases, data, CONSTANTS)
    FIXOM_df = calculate_FIXOM(Cases, CAPEX_df, data, CONSTANTS)
    VAROM_df = calculate_VAROM(Cases, CAPEX_df, data, CONSTANTS)
    Energy_df = calculate_energy(Cases, data, CONSTANTS)
    ETS_df = calculate_ETS(Cases, data, CONSTANTS)
    HRS_df = calculate_HRS(Cases, data, CONSTANTS)
    BET_CH_df, FF_CH_df = calculate_BET_charging(Cases, data, CONSTANTS)

    # Debugging: Check the type of each component
    #print("CAPEX_df type:", type(CAPEX_df))
    #print("FIXOM_df type:", type(FIXOM_df))
    #print("VAROM_df type:", type(VAROM_df))
    #print("Energy_df type:", type(Energy_df))
    #print("ETS_df type:", type(ETS_df))
    #print("HRS_df type:", type(HRS_df))
    #print("BET_CH_df type:", type(BET_CH_df))
    #print("FF_CH_df type:", type(FF_CH_df))

    # Ensure all components are DataFrames
    if isinstance(CAPEX_df, list):
        CAPEX_df = pd.DataFrame(CAPEX_df)
    if isinstance(FIXOM_df, list):
        FIXOM_df = pd.DataFrame(FIXOM_df)
    if isinstance(VAROM_df, list):
        VAROM_df = pd.DataFrame(VAROM_df)
    if isinstance(Energy_df, list):
        Energy_df = pd.DataFrame(Energy_df)
    if isinstance(ETS_df, list):
        ETS_df = pd.DataFrame(ETS_df)
    if isinstance(HRS_df, list):
        HRS_df = pd.DataFrame(HRS_df)
    if isinstance(BET_CH_df, list):
        BET_CH_df = pd.DataFrame(BET_CH_df)
    if isinstance(FF_CH_df, list):
        FF_CH_df = pd.DataFrame(FF_CH_df)

    # Concatenate all DataFrames
    Total_Matrix = pd.concat([
        CAPEX_df.reset_index(drop=True),
        FIXOM_df.reset_index(drop=True),
        VAROM_df.reset_index(drop=True),
        Energy_df.reset_index(drop=True),
        ETS_df.reset_index(drop=True),
        HRS_df.reset_index(drop=True),
        BET_CH_df.reset_index(drop=True),
        FF_CH_df.reset_index(drop=True)
    ], axis=0)

    # Standardize scenario names
    Total_Matrix.loc[Total_Matrix['Scenario'] == 'Medium', 'Scenario'] = 'Average'
    Total_Matrix.loc[Total_Matrix['Scenario'] == 'NZ', 'Scenario'] = 'Average'
    Total_Matrix.loc[Total_Matrix['Scenario'] == 'All', 'Scenario'] = 'Average'

    # Fill missing scenarios
    Total_Matrix_filled = fill_missing_scenarios(Total_Matrix)
    return Total_Matrix_filled

def Total_Matrix_km(Cases, data, CONSTANTS):
    Total_Matrix=make_Total_matrix(Cases, data, CONSTANTS)
    Cases_2 = Total_Matrix.Case.drop_duplicates().tolist()
    Life=CONSTANTS["Life"]
    working_days=CONSTANTS["working_days"]
    for case in Cases_2:
        km = Cases[Cases["Case"] == case]["Distance"].iat[0] * working_days
        life_km= km*Life

        # Define mask for BET CAPEX and FIXOM components
        mask = (Total_Matrix["Case"] == case)

        # Compute new values
        values_v = Total_Matrix["Value"].copy()  # Copy values to avoid modifying original dataframe directly
        values_v[mask] /= (life_km) # Apply specific division for BET CAPEX & FIXOM

        # Assign back to DataFrame
        Total_Matrix["Value"] = values_v
    return Total_Matrix
