#%%
import pandas as pd

df = pd.read_csv(r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Runs\MeshGraphNet_20260702_11_fin\test_case_csv_outputs_vonmises_11_fin\test_case_metrics_summary.csv")

df.head()
# %%
df.plot(x="case_name", y=["rmse"], kind="bar", figsize=(12, 6), title="RMSE Stress Predictions by Test Case")

# %%
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"
fig = px.bar(df, x="case_name", y="rmse", title="RMSE Stress Predictions by Test Case")
fig.show()
fig.write_html(r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Runs\MeshGraphNet_20260702_11_fin\test_case_csv_outputs_vonmises_11_fin\rmse_stress_predictions_by_test_case.html")
# %%
# tha DoE cases with the highest RMSE are:
#  8.2021,19.518,2.3635,0.898,-0.3314,19.1272,Alu,BC1,17

#  11.7973,17.9557,2.2684,0.6538,0.6549,17.6225,Steel,BC1,38

#  11.4316,20.8228,2.16,-0.8913,0.9464,13.2006,Steel,BC1,81

#%%
df1 = pd.read_csv(r'D:\Agentic_AI\Plate_with_a_hole\Step4_Refinement_plate_model\DoE_data\test_set.csv')

df1.head()

# %%

fig = px.scatter(df1, x=df1.index, y=["length", 'breadth'], title="Comparison",
             facet_col="Material",
             custom_data=["Material", "bc_loc", "Load",'Cirlce_radius', 'Circle_pos_X', 'Circle_pos_Y'],
             #hover_data=["length", "breadth"],
             #hover_data={"length": True, "breadth": True, "Material": True, "bc_loc": True, "Load": True},
             )
fig.update_layout(hovermode="x unified")
fig.update_traces(
    hovertemplate=
    "index: %{x}<br>"
    "value: %{y}<br>"
    "variable: %{fullData.name}<br>"
    "Material: %{customdata[0]}<br>"
    "bc_loc: %{customdata[1]}<br>"
    "Load: %{customdata[2]}<br>"
    "Circle_radius: %{customdata[3]}<br>"
    "Circle_pos_X: %{customdata[4]}<br>"
    "Circle_pos_Y: %{customdata[5]}<extra></extra>"
)
fig.show()

# %%
df3 = pd.read_csv(r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\DoE_data\test_set.csv")
df3.head()

# %%
df4 = pd.read_csv(r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Runs\MeshGraphNet_20260702_11_fin\test_case_csv_outputs_vonmises_11_fin\test_case_metrics_summary.csv")
df4.head()
counter = [int(name.split('_')[1]) for name in df4['case_name']]
df4["counter"] = counter

# %%
df5 = pd.merge(df3, df4, on="counter", how="inner")

#%%
temp = 10 + 1000 * (df5["Load"] - df5["Load"].min()) / (df5["Load"].max() - df5["Load"].min())

#%%
fig = px.scatter(df5, x=df5.index, y=["length", 'breadth'], title="Comparison",
             facet_col="Material",
             #color="rmse",
             size=temp,
             symbol="bc_loc",
             custom_data=["Material", "bc_loc", "Load",'Cirlce_radius', 'Circle_pos_X', 'Circle_pos_Y', "mae","rmse", "r2","case_name"],
             #hover_data=["length", "breadth"],
             #hover_data={"length": True, "breadth": True, "Material": True, "bc_loc": True, "Load": True},
             )
fig.update_layout(hovermode="x unified")
fig.update_traces(
    hovertemplate=
    "index: %{x}<br>"
    "value: %{y}<br>"
    "length: %{fullData.name}<br>"
    "Material: %{customdata[0]}<br>"
    "bc_loc: %{customdata[1]}<br>"
    "Load: %{customdata[2]}<br>"
    "Circle_radius: %{customdata[3]}<br>"
    "Circle_pos_X: %{customdata[4]}<br>"
    "Circle_pos_Y: %{customdata[5]}<br>"
    "MAE: %{customdata[6]:.2f}<br>"
    "RMSE: %{customdata[7]:.2f}<br>"
    "R2: %{customdata[8]:.2f}<br>"
    "case_name: %{customdata[9]}<extra></extra>"
)
fig.show()
fig.write_html(r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Runs\MeshGraphNet_20260702_11_fin\test_case_csv_outputs_vonmises_11_fin\DoE_test_results_summary.html")


# %%
