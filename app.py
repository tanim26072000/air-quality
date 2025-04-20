import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from sklearn.metrics import mean_absolute_error

# ------------- Page & Title Setup -------------
st.set_page_config(page_title="Interactive PM2.5 Data", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center;'>
        Interactive PM<sub>2.5</sub> Data Visualization with Spatio-temporal Selection
    </h1>
    """,
    unsafe_allow_html=True,
)

# Define available months
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Apply custom CSS for selectboxes
st.markdown(
    """
    <style>
    .stSelectbox select {
        width: fit-content !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------- User Inputs -------------

col1, col2 = st.columns(2)
with col1:
    selected_month = st.selectbox("Select a month", months, key="select_month")
with col2:
    selected_year = st.selectbox("Select a year", list(
        range(2000, 2020)), key="select_year")

# Load data from CSV file
data = pd.read_csv(f"data/{selected_month}_{selected_year}.csv", compression="gzip",    # tell pandas it’s gzipped
                   header=0,              # adjust if you have a header row
                   sep=",")

# Scope selection
scope = st.radio(
    "Select Map Scope",
    ("Whole Bangladesh", "Division", "District"),
    horizontal=True,
)
if scope == "Whole Bangladesh":
    filtered_geo_data = data
    s = "Bangladesh"
    z = 5
elif scope == "Division":
    divs = sorted(data["ADM1_EN"].unique())
    selected_division = st.selectbox(
        "Select Division", divs, key="select_division")
    s = selected_division
    z = 7.5
    filtered_geo_data = data[data["ADM1_EN"] == selected_division]
elif scope == "District":
    dists = sorted(data["ADM2_EN"].unique())
    selected_district = st.selectbox(
        "Select District", dists, key="select_district")
    s = selected_district
    z = 8
    filtered_geo_data = data[data["ADM2_EN"] == selected_district]

# Model selections
logic_help = """
Here’s what you’ll see when you pick two models:

1. **Same & Observed** → 1 map (just the Observed data).  
2. **Same & not Observed** → 2 maps (Observed baseline + your selected model).  
3. **Different & one is Observed** → 2 maps (Observed + your other model).  
4. **Different & neither is Observed** → 3 maps (Observed + both chosen models).
"""

col1, col2 = st.columns(2)
with col1:
    model1 = st.selectbox("Select first model",
                          ["Observed", "GNN+LSTM", "GNN", "CNN+LSTM", "CNN"],
                          key="select_model", help= logic_help)
with col2:
    model2 = st.selectbox("Select second model",
                          ["Observed", "GNN+LSTM", "GNN", "CNN+LSTM", "CNN"],
                          key="select_model_2", help=logic_help)

# Normalize to lowercase for column references.
m1 = model1.lower()
m2 = model2.lower()

# Subset the data; use only the needed columns.
df = filtered_geo_data[
    [
        "latitude",
        "longitude",
        "observed",
        "gnn+lstm",
        "gnn",
        "cnn+lstm",
        "cnn",
        "ADM1_EN",  # Division
        "ADM2_EN",  # District
        "ADM3_EN",  # Upazila
    ]
]
main_df = df.copy()

# ------------- Helper Functions -------------


def get_status_color(value):
    """Return a tuple (status, [R,G,B,A], cssColor) based on PM2.5 value."""
    if value <= 12:
        return "Good", [0, 255, 0, 160], "green"
    elif value <= 35.4:
        return "Moderate", [255, 255, 0, 160], "yellow"
    elif value <= 55.4:
        return "Unhealthy for Sensitive Groups", [255, 165, 0, 160], "orange"
    elif value <= 150.4:
        return "Unhealthy", [255, 0, 0, 160], "red"
    elif value <= 250.4:
        return "Very Unhealthy", [153, 50, 204, 160], "purple"
    else:
        return "Hazardous", [128, 0, 0, 160], "darkred"


def get_stats(series):
    """Return a dictionary with the Min, Max, and Mean values."""
    return {"Min": f"{series.min():.2f}", "Max": f"{series.max():.2f}", "Mean": f"{series.mean():.2f}"}


def get_aq_counts(series):
    """Return counts for each air quality category."""
    categories = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    status = series.apply(lambda x: get_status_color(x)[0])
    counts = status.value_counts().to_dict()
    for cat in categories:
        counts.setdefault(cat, 0)
    return counts


# Compute additional fields for tooltips and visualization.
df = df.assign(
    observed_status=df["observed"].apply(lambda x: get_status_color(x)[0]),
    observed_color=df["observed"].apply(lambda x: get_status_color(x)[1]),
    observed_color_css=df["observed"].apply(lambda x: get_status_color(x)[2]),
    predicted_status_1=df[m1].apply(lambda x: get_status_color(x)[0]),
    predicted_status_2=df[m2].apply(lambda x: get_status_color(x)[0]),
    predicted_color_1=df[m1].apply(lambda x: get_status_color(x)[1]),
    predicted_color_2=df[m2].apply(lambda x: get_status_color(x)[1]),
    predicted_color_css_1=df[m1].apply(lambda x: get_status_color(x)[2]),
    predicted_color_css_2=df[m2].apply(lambda x: get_status_color(x)[2]),
    predicted_elevation_1=df[m1] * 50,
    predicted_elevation_2=df[m2] * 50,
    observed_elevation=df["observed"] * 50,
)

# ------------- Dynamic Tooltip Generator Functions -------------


def generate_plotly_tooltip(num_layers):
    """
    Returns a dynamic Plotly hovertemplate based on the number of layers:
      1 layer: only observed.
      2 layers: observed + one predicted.
      3 layers: observed + two predicted.
      
    Expected custom_data order:
       Indices 0-7: latitude, longitude, ADM1_EN, ADM2_EN, ADM3_EN, observed, observed_status, observed_color_css
       For predicted layer 1: indices 8, 9, 10 (predicted value, predicted status, predicted color)
       For predicted layer 2 (if applicable): indices 11, 12, 13
    """
    tooltip = (
        "<b>Latitude:</b> %{customdata[0]} <br>"
        "<b>Longitude:</b> %{customdata[1]} <br>"
        "<b>Division:</b> %{customdata[2]} <br>"
        "<b>District:</b> %{customdata[3]} <br>"
        "<b>Upazila:</b> %{customdata[4]} <br>"
        "<b>Observed:</b> <span style='color:%{customdata[7]};'>%{customdata[5]} (%{customdata[6]})</span><br>"
    )
    if num_layers == 1:
        tooltip += "<extra></extra>"
    elif num_layers == 2:
        tooltip += (
            f"<b>{m2.upper() if m1== 'observed' else m1.upper()}:</b> <span style='color:%{{customdata[10]}};'>"
            "%{customdata[8]} (%{customdata[9]})</span><extra></extra>"
        )
    elif num_layers == 3:
        tooltip += (
            f"<b>{m1.upper()}:</b> <span style='color:%{{customdata[10]}};'>"
            "%{customdata[8]} (%{customdata[9]})</span><br>"
            f"<b>{m2.upper()}:</b> <span style='color:%{{customdata[13]}};'>"
            "%{customdata[11]} (%{customdata[12]})</span><extra></extra>"
        )
    return tooltip


def generate_pydeck_tooltip(num_layers, primary_model, secondary_model=None):
    """
    Returns a dynamic Pydeck tooltip HTML string.
    
    Expected attributes in the data:
      - Base: latitude, longitude, ADM1_EN, ADM2_EN, ADM3_EN, observed, observed_status, observed_color_css
      - For predicted:
          If 2 layers: use either predicted_color_css_1 and predicted_status_1 if primary_model != "observed",
                     or predicted_color_css_2 and predicted_status_2 if the observed model is primary.
          If 3 layers: uses both predicted_x attributes.
    """
    tooltip = (
        "<b>Latitude:</b> {latitude} <br>"
        "<b>Longitude:</b> {longitude} <br>"
        "<b>Division:</b> {ADM1_EN} <br>"
        "<b>District:</b> {ADM2_EN} <br>"
        "<b>Upazila:</b> {ADM3_EN} <br>"
        "<b>Observed:</b> <span style='color:{observed_color_css};'>"
        "{observed} ({observed_status})</span><br>"
    )
    if num_layers == 1:
        return tooltip
    elif num_layers == 2:
        # In two-layer case, if observed is one model, then the predicted values come from the other.
        # Determine which predicted field to use.
        tooltip += (
            f"<b>{primary_model.upper()}:</b> <span style='color:{{predicted_color_css_1}};'>"
            f"{{{primary_model.lower()}}} ({{predicted_status_1}})</span>"
        )
        return tooltip
    elif num_layers == 3:
        tooltip += (
            f"<b>{m1.upper()}:</b> <span style='color:{{predicted_color_css_1}};'>"
            f"{{{primary_model.lower()}}} ({{predicted_status_1}})</span><br>"
            f"<b>{m2.upper()}:</b> <span style='color:{{predicted_color_css_2}};'>"
            f"{{{secondary_model.lower()}}} ({{predicted_status_2}})</span>"
        )
        return tooltip


# ------------- Preparing Custom Data & Dynamic Tooltip for Plotly -------------
# Base custom data (observed values)
base_data = [
    "latitude",
    "longitude",
    "ADM1_EN",  # Division
    "ADM2_EN",  # District
    "ADM3_EN",  # Upazila
    "observed",
    "observed_status",
    "observed_color_css"
]

# Determine the number of layers and set up the custom_data array.
# Logic:
#  - If models are the same:
#       If "observed": one layer (num_layers = 1)
#       Else: two layers (num_layers = 2) using predicted from that model.
#  - If models differ:
#       If one of them is "observed": two layers (num_layers = 2) with the non-observed model as predicted.
#       Else: three layers (num_layers = 3) with both predicted models.
if m1 == m2:
    if m1 == "observed":
        num_layers = 1
        custom_data = base_data
    else:
        num_layers = 2
        custom_data = base_data + \
            [m1, "predicted_status_1", "predicted_color_css_1"]
else:
    if "observed" in [m1, m2]:
        num_layers = 2
        if m1 == "observed":
            custom_data = base_data + \
                [m2, "predicted_status_2", "predicted_color_css_2"]
        else:
            custom_data = base_data + \
                [m1, "predicted_status_1", "predicted_color_css_1"]
    else:
        num_layers = 3
        custom_data = base_data + [m1, "predicted_status_1", "predicted_color_css_1",
                                   m2, "predicted_status_2", "predicted_color_css_2"]

plotly_tooltip = generate_plotly_tooltip(num_layers)


def make_px_fig(color_col, title_suffix, tooltip_text):
    """Create a Plotly scatter figure with the custom data and dynamic tooltip."""
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        color_discrete_map={
            "Good": "green",
            "Moderate": "yellow",
            "Unhealthy for Sensitive Groups": "orange",
            "Unhealthy": "red",
            "Very Unhealthy": "purple",
            "Hazardous": "darkred"
        },
        zoom=z,
        center={
            "lat": df["latitude"].mean(),    # center on your data
            "lon": df["longitude"].mean()
        },
        mapbox_style="carto-positron",         # your base‑map
        custom_data=custom_data,
        height=600,
        opacity=0.8,
    )
    fig.update_traces(hovertemplate=tooltip_text)
    fig.update_layout(mapbox_style="carto-positron",
                      margin={"l": 0, "r": 0, "t": 0, "b": 0}, 
                      showlegend=False)
    return fig


# ------------- Dynamic Pydeck Tooltip Setup -------------
# For Pydeck, determine the tooltip based on the same num_layers.
# In two-layer case, pick the predicted fields based on which model is not "observed".
if num_layers == 1:
    pydeck_tooltip = generate_pydeck_tooltip(num_layers, None)
elif num_layers == 2:
    if m1 == "observed":
        primary_model = m2
    else:
        primary_model = m1
    pydeck_tooltip = generate_pydeck_tooltip(num_layers, primary_model)
elif num_layers == 3:
    pydeck_tooltip = generate_pydeck_tooltip(num_layers, m1, m2)

# ------------- Generate Plot & Tables -------------
if st.button("Generate Plot"):
    if scope == "Whole Bangladesh":
        # Use Plotly for Whole Bangladesh with multi-plot logic.
        plots = []    # List to store (title, figure) tuples.
        table_columns = {}
        aq_columns = {}
        # When both model choices are identical.
        if m1 == m2:
            if m1 == "observed":
                # Only one Plotly map is generated.
                fig_obs = make_px_fig(
                    "observed_status", "Observed Data", plotly_tooltip)
                plots.append(("Observed Data", fig_obs))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
            else:
                # If same but not "observed", generate two maps:
                # one for observed and one for the common predicted layer.
                fig_obs = make_px_fig(
                    "observed_status", "Observed Data", plotly_tooltip)
                fig_model = make_px_fig(
                    "predicted_status_1", f"Data Map by {m1.upper()}", plotly_tooltip)
                plots.append(("Observed Data", fig_obs))
                plots.append((f"Data Map by {m1.upper()}", fig_model))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
                table_columns[m1.upper()] = get_stats(df[m1])
                aq_columns[m1.upper()] = get_aq_counts(df[m1])
        else:
            # When the two model choices differ.
            if "observed" in [m1, m2]:
                if m1 == "observed":
                    non_obs = m2
                    fig_obs = make_px_fig(
                        "observed_status", "Observed Data", plotly_tooltip)
                    fig_model = make_px_fig(
                        "predicted_status_2", f"Data Map by {non_obs.upper()}", plotly_tooltip)
                else:
                    non_obs = m1
                    fig_obs = make_px_fig(
                        "observed_status", "Observed Data", plotly_tooltip)
                    fig_model = make_px_fig(
                        "predicted_status_1", f"Data Map by {non_obs.upper()}", plotly_tooltip)
                plots.append(("Observed Data", fig_obs))
                plots.append((f"Data Map by {non_obs.upper()}", fig_model))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
                table_columns[non_obs.upper()] = get_stats(df[non_obs])
                aq_columns[non_obs.upper()] = get_aq_counts(df[non_obs])
            else:
                # If neither is observed; generate three maps.
                fig_obs = make_px_fig(
                    "observed_status", "Observed Data", plotly_tooltip)
                fig_m1 = make_px_fig(
                    "predicted_status_1", f"Data Map by {m1.upper()}", plotly_tooltip)
                fig_m2 = make_px_fig(
                    "predicted_status_2", f"Data Map by {m2.upper()}", plotly_tooltip)
                plots.append(("Observed Data", fig_obs))
                plots.append((f"Data Map by {m1.upper()}", fig_m1))
                plots.append((f"Data Map by {m2.upper()}", fig_m2))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
                table_columns[m1.upper()] = get_stats(df[m1])
                aq_columns[m1.upper()] = get_aq_counts(df[m1])
                table_columns[m2.upper()] = get_stats(df[m2])
                aq_columns[m2.upper()] = get_aq_counts(df[m2])

        # Render the Plotly figures side by side.
        num_plots = len(plots)
        cols = st.columns(num_plots)
        for i, (title, fig) in enumerate(plots):
            with cols[i]:
                st.markdown(
                    f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
        # Additional code for descriptive statistics and CSV download can follow here.

    else:
        # For Division and District scopes, use Pydeck.
        view_state = pdk.ViewState(
            latitude=df["latitude"].mean(),
            longitude=df["longitude"].mean(),
            zoom=z,
            pitch=40,
        )
        plots = []
        table_columns = {}
        aq_columns = {}
        # When both model choices are identical.
        if m1 == m2:
            if m1 == "observed":
                # Only one plot.
                observed_layer = pdk.Layer(
                    "GridCellLayer",
                    df,
                    get_position="[longitude, latitude]",
                    cell_size=1000,
                    get_elevation="observed_elevation",
                    get_fill_color="observed_color",
                    pickable=True,
                    extruded=True,
                    elevation_scale=1,
                )
                deck_obs = pdk.Deck(
                    layers=[observed_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                plots.append(("Observed Data", deck_obs))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
            else:
                # When same but not observed, generate two plots: observed and predicted.
                observed_layer = pdk.Layer(
                    "GridCellLayer",
                    df,
                    get_position="[longitude, latitude]",
                    cell_size=1000,
                    get_elevation="observed_elevation",
                    get_fill_color="observed_color",
                    pickable=True,
                    extruded=True,
                    elevation_scale=1,
                )
                deck_obs = pdk.Deck(
                    layers=[observed_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                predicted_layer = pdk.Layer(
                    "GridCellLayer",
                    df,
                    get_position="[longitude, latitude]",
                    cell_size=1000,
                    get_elevation="predicted_elevation_1",
                    get_fill_color="predicted_color_1",
                    pickable=True,
                    extruded=True,
                    elevation_scale=1,
                )
                deck_model = pdk.Deck(
                    layers=[predicted_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                plots.append(("Observed Data", deck_obs))
                plots.append((f"Data Map by {m1.upper()}", deck_model))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
                table_columns[m1.upper()] = get_stats(df[m1])
                aq_columns[m1.upper()] = get_aq_counts(df[m1])
        else:
            # When model choices differ.
            if "observed" in [m1, m2]:
                if m1 == "observed":
                    non_obs = m2
                    observed_layer = pdk.Layer(
                        "GridCellLayer",
                        df,
                        get_position="[longitude, latitude]",
                        cell_size=1000,
                        get_elevation="observed_elevation",
                        get_fill_color="observed_color",
                        pickable=True,
                        extruded=True,
                        elevation_scale=1,
                    )
                    predicted_layer = pdk.Layer(
                        "GridCellLayer",
                        df,
                        get_position="[longitude, latitude]",
                        cell_size=1000,
                        get_elevation="predicted_elevation_2",
                        get_fill_color="predicted_color_2",
                        pickable=True,
                        extruded=True,
                        elevation_scale=1,
                    )
                else:
                    non_obs = m1
                    observed_layer = pdk.Layer(
                        "GridCellLayer",
                        df,
                        get_position="[longitude, latitude]",
                        cell_size=1000,
                        get_elevation="observed_elevation",
                        get_fill_color="observed_color",
                        pickable=True,
                        extruded=True,
                        elevation_scale=1,
                    )
                    predicted_layer = pdk.Layer(
                        "GridCellLayer",
                        df,
                        get_position="[longitude, latitude]",
                        cell_size=1000,
                        get_elevation="predicted_elevation_1",
                        get_fill_color="predicted_color_1",
                        pickable=True,
                        extruded=True,
                        elevation_scale=1,
                    )
                deck_obs = pdk.Deck(
                    layers=[observed_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                deck_model = pdk.Deck(
                    layers=[predicted_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                plots.append(("Observed Data", deck_obs))
                plots.append((f"Data Map by {non_obs.upper()}", deck_model))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
                table_columns[non_obs.upper()] = get_stats(df[non_obs])
                aq_columns[non_obs.upper()] = get_aq_counts(df[non_obs])
            else:
                # Neither model is observed; generate three plots.
                fig_obs_layer = pdk.Layer(
                    "GridCellLayer",
                    df,
                    get_position="[longitude, latitude]",
                    cell_size=1000,
                    get_elevation="observed_elevation",
                    get_fill_color="observed_color",
                    pickable=True,
                    extruded=True,
                    elevation_scale=1,
                )
                deck_obs = pdk.Deck(
                    layers=[fig_obs_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                layer_m1 = pdk.Layer(
                    "GridCellLayer",
                    df,
                    get_position="[longitude, latitude]",
                    cell_size=1000,
                    get_elevation="predicted_elevation_1",
                    get_fill_color="predicted_color_1",
                    pickable=True,
                    extruded=True,
                    elevation_scale=1,
                )
                deck_m1 = pdk.Deck(
                    layers=[layer_m1],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                layer_m2 = pdk.Layer(
                    "GridCellLayer",
                    df,
                    get_position="[longitude, latitude]",
                    cell_size=1000,
                    get_elevation="predicted_elevation_2",
                    get_fill_color="predicted_color_2",
                    pickable=True,
                    extruded=True,
                    elevation_scale=1,
                )
                deck_m2 = pdk.Deck(
                    layers=[layer_m2],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/streets-v12",
                    tooltip={"html": pydeck_tooltip,
                             "style": {"color": "white"}},
                )
                plots.append(("Observed Data", deck_obs))
                plots.append((f"Data Map by {m1.upper()}", deck_m1))
                plots.append((f"Data Map by {m2.upper()}", deck_m2))
                table_columns["OBSERVED"] = get_stats(df["observed"])
                aq_columns["OBSERVED"] = get_aq_counts(df["observed"])
                table_columns[m1.upper()] = get_stats(df[m1])
                aq_columns[m1.upper()] = get_aq_counts(df[m1])
                table_columns[m2.upper()] = get_stats(df[m2])
                aq_columns[m2.upper()] = get_aq_counts(df[m2])
        num_plots = len(plots)
        cols = st.columns(num_plots)
        for i, (title, deck_obj) in enumerate(plots):
            with cols[i]:
                st.markdown(
                    f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
                st.pydeck_chart(deck_obj)
        # ------------- Statistics and Model Comparison Information -------------

    # Assume that m1 and m2 hold the lower-case model names, where "observed" is the
    # column with the actual data. We now determine which predicted models (if any) to compare.
    predicted_list = []
    if m1 != "observed":
        if m1 not in predicted_list:
            predicted_list.append(m1)
    if m2 != "observed":
        if m2 not in predicted_list:
            predicted_list.append(m2)

    # Final table columns: always include observed as the first column,
    # then add one column for each predicted model.
    table_columns = ["observed"] + predicted_list

    # Compute total data points:
    total_count = len(df)

    # Initialize a dictionary to hold the statistical rows.
    stats = {
        "Statistic/Category": [
            "Minimum (µg/m³)",
            "Maximum (µg/m³)",
            "Mean (µg/m³)",
            "<span style='color:green;'>Good</span> count",
            "<span style='color:yellow;'>Moderate</span> count",
            "<span style='color:orange;'>Unhealthy for Sensitive Groups</span> count",
            "<span style='color:red;'>Unhealthy</span> count",
            "<span style='color:purple;'>Very Unhealthy</span> count",
            "<span style='color:darkred;'>Hazardous</span> count"
        ]
    }

    # For each column (observed and each predicted), compute basic stats and AQ counts.
    for col in table_columns:
        # Compute basic numeric stats.
        min_val = f"{df[col].min():.2f}"
        max_val = f"{df[col].max():.2f}"
        mean_val = f"{df[col].mean():.2f}"
        # Compute air quality counts using the provided function.
        counts = get_aq_counts(df[col])
        stats[col.upper()] = [
            min_val,
            max_val,
            mean_val,
            counts.get("Good", 0),
            counts.get("Moderate", 0),
            counts.get("Unhealthy for Sensitive Groups", 0),
            counts.get("Unhealthy", 0),
            counts.get("Very Unhealthy", 0),
            counts.get("Hazardous", 0)
        ]

    # Create the DataFrame for the statistics.
    stats_df = pd.DataFrame(stats)
    stats_df.rename(
        columns={"OBSERVED": "Observed"}, inplace=True)

    # Now, create a second DataFrame for error metrics.
    error_data = {"Error Metric": ["RMSE", "MAE"]}
    # For every predicted model column, compare it with the observed column.
    for col in predicted_list:
        rmse_val = np.sqrt(((df["observed"] - df[col]) ** 2).mean())
        mae_val = mean_absolute_error(df["observed"], df[col])
        error_data[f"{col.upper()}"] = [
            f"{rmse_val:.2f}", f"{mae_val:.2f}"]

    error_df = pd.DataFrame(error_data)

    # Display the results.
    st.write(f"**Total data points:** {total_count}")
    st.write("### Statistical Overview")

    # Convert the DataFrame to an HTML table with escaping disabled.
    html_table = stats_df.to_html(escape=False, index=False)

    # Display the HTML table in Streamlit.
    st.markdown(html_table, unsafe_allow_html=True)

    if predicted_list:  # Only show error metrics if there are predicted models.
        st.write("### Error Metrics for Predicted Models")
        st.dataframe(error_df, hide_index=True)
    st.markdown(
        f"""
    ### PM$_{{2.5}}$ Concentration Ranges and Health Impacts:
    - <span style='color:green;'>**0-12 µg/m³ (Good)**</span>: Satisfactory, posing little or no health risk.<br>
    - <span style='color:yellow;'>**12.1-35.4 µg/m³ (Moderate)**</span>: Acceptable, but sensitive individuals may experience health issues.<br>
    - <span style='color:orange;'>**35.5-55.4 µg/m³ (Unhealthy for Sensitive Groups)**</span>: Sensitive groups may experience health effects.<br>
    - <span style='color:red;'>**55.5-150.4 µg/m³ (Unhealthy)**</span>: Everyone may experience health effects.<br>
    - <span style='color:purple;'>**150.5-250.4 µg/m³ (Very Unhealthy)**</span>: Severe health effects for everyone.<br>
    - <span style='color:darkred;'>**> 250.5 µg/m³ (Hazardous)**</span>: Health warnings; emergency conditions.<br><br>
    For more information, visit [here](https://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/).
    """, unsafe_allow_html=True)

    # ------------------ Data Download Button ------------------
    # Rename columns in main_df to be more user-friendly before downloading.
    main_df.rename(
        columns={'ADM1_EN': 'Division',
                 'ADM2_EN': 'District', 'ADM3_EN': 'Upazilla'},
        inplace=True
    )
    csv_data = main_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name=f"PM2.5_{s}_{selected_month}_{selected_year}.csv",
        mime='text/csv'
    )
