# 1. Import libraries

from pathlib import Path
import base64
import json
import os
import re
import time
import html
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow.parquet as pq

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover
    OllamaClient = None

try:
    import pydeck as pdk
except ImportError:  # pragma: no cover
    pdk = None

try:
    from streamlit_searchbox import st_searchbox
except ImportError:  # pragma: no cover
    st_searchbox = None


# 2. Define file path and selected columns

APP_DIR = Path(__file__).resolve().parent
DOWNLOAD_ICON_PATH = APP_DIR / "assets" / "download_icon.png"
DOWNLOAD_ICON_DATA_URI = (
    "data:image/png;base64,"
    + base64.b64encode(DOWNLOAD_ICON_PATH.read_bytes()).decode("ascii")
)


def image_data_uri(path: Path) -> str:
    """Encode a local image for reliable use in dashboard HTML."""
    suffix = path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    return (
        f"data:{mime_type};base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
    )


NTNU_LOGO_DATA_URI = image_data_uri(APP_DIR / "assets" / "logo_ntnu_transparent.png")
TYNDALL_LOGO_DATA_URI = image_data_uri(APP_DIR / "assets" / "logo_tyndall_transparent.png")
AMBS_LOGO_DATA_URI = image_data_uri(APP_DIR / "assets" / "logo_ambs_transparent.png")
DATA_PATH_CANDIDATES = [
    APP_DIR / "data" / "voyages_enriched.parquet",
    APP_DIR / "voyages_enriched.parquet",
    Path.home()
    / "Documents"
    / "ERP"
    / "NorthSea_Emissions_Dashboard_GitHub"
    / "data"
    / "voyages_enriched.parquet",
]
DEFAULT_PARQUET_PATH = str(
    next(
        (path for path in DATA_PATH_CANDIDATES if path.exists()),
        DATA_PATH_CANDIDATES[0],
    )
)

ANALYSIS_START_YEAR = 2018
ANALYSIS_END_YEAR = 2021

APPROVED_DATASETS = {
    "North Sea Enriched Voyages": {
        "path": DEFAULT_PARQUET_PATH,
        "description": "Voyage, vessel, port, route, and emissions data",
    }
}

COLUMNS_TO_LOAD = [
    "imo",
    "mmsi",
    "vessel_name",
    "vessel_general_type",
    "vessel_detailed_type",
    "vessel_flag_2024",
    "vessel_year_of_build",
    "vessel_age_at_departure",
    "vessel_age_group",
    "vessel_size_group_dwt",
    "vessel_gross_tonnage",
    "vessel_deadweight_tonnage",
    "vessel_engine_power",
    "country_o",
    "country_d",
    "port_id_o",
    "port_id_d",
    "origin_port_display",
    "destination_port_display",
    "port_route_display",
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
    "departure_time",
    "arrival_time",
    "departure_year",
    "departure_month",
    "voyage_scope",
    "route_directional",
    "country_pair",
    "ship_type_clean",
    "delta_dist_km",
    "delta_time_s",
    "dwt",
    "CO2",
    "CO2_ME_kg",
    "CO2_AE_kg",
    "CO2_boiler_kg",
    "CO2_per_km",
    "CO2_per_dwt_km",
    "ME_CO2_share",
    "AE_CO2_share",
    "boiler_CO2_share",
]

COUNTRY_NAME_MAP = {
    "BEL": "Belgium",
    "CAN": "Canada",
    "CHN": "China",
    "CYP": "Cyprus",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "IRL": "Ireland",
    "ISL": "Iceland",
    "ITA": "Italy",
    "LTU": "Lithuania",
    "LVA": "Latvia",
    "MLT": "Malta",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "POL": "Poland",
    "PRT": "Portugal",
    "RUS": "Russia",
    "SWE": "Sweden",
    "USA": "United States",
}

CLIMATE_COLORS = {
    "background": "#F4F6F1",
    "surface": "#FFFFFF",
    "surface_alt": "#E8ECE1",
    "text": "#1F2A24",
    "muted": "#657369",
    "border": "#D7E0D5",
    "navy": "#1F4D3A",
    "teal": "#61743A",
    "green": "#6A7E40",
    "amber": "#D9A441",
    "coral": "#D97845",
    "blue": "#2F6F73",
    "purple": "#756B9A",
}

CLIMATE_SEQUENCE = [
    CLIMATE_COLORS["coral"],
    CLIMATE_COLORS["teal"],
    CLIMATE_COLORS["navy"],
    CLIMATE_COLORS["amber"],
    CLIMATE_COLORS["green"],
    CLIMATE_COLORS["blue"],
    CLIMATE_COLORS["purple"],
]

# Use a web-safe family first so the host page, iframe components and charts
# render with the same typeface on macOS, Windows and Linux.
PREMIUM_FONT_STACK = "Arial, Helvetica, sans-serif"


# 3. Configure Streamlit page layout

st.set_page_config(
    page_title="North Sea Shipping Emissions Explorer",
    layout="wide",
)

st.markdown(
    """
    <style>
    .selection-summary {
        margin: 0.7rem 0 1.35rem 0;
        padding: 1rem 1.15rem;
        border: 1px solid #d7e1f0;
        border-left: 5px solid #2b6cb0;
        border-radius: 8px;
        background: linear-gradient(90deg, #f6f9fe 0%, #ffffff 100%);
    }
    .selection-kicker {
        margin-bottom: 0.55rem;
        color: #526072;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .selection-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .selection-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.38rem 0.62rem;
        border: 1px solid #c9d8ea;
        border-radius: 999px;
        background: #ffffff;
        color: #253247;
        font-size: 0.92rem;
        font-weight: 600;
        line-height: 1.2;
        box-shadow: 0 1px 2px rgba(30, 41, 59, 0.06);
    }
    .dashboard-partner-footer {
        margin: 3rem 0 0.75rem 0;
        padding: 1rem 1.15rem 0.9rem;
        border-top: 1px solid #D9E2D8;
        color: #66736C;
    }
    .dashboard-footer-logos {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.7rem;
    }
    .dashboard-footer-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 118px;
        height: 42px;
        overflow: hidden;
        background: transparent;
    }
    .dashboard-footer-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center;
    }
    .dashboard-footer-logo.logo-ntnu img { object-position: 50% 49%; }
    .dashboard-footer-logo.logo-tyndall img { object-position: 50% 51%; }
    .dashboard-footer-logo.logo-ambs img { object-position: 50% 45%; }
    .dashboard-footer-source {
        margin: 0;
        text-align: right;
        font-size: 0.76rem;
        line-height: 1.45;
    }
    .dashboard-footer-source strong {
        color: #244F3F;
        font-weight: 700;
    }
    @media (max-width: 680px) {
        .dashboard-footer-logos {
            justify-content: flex-start;
            gap: 0.55rem;
        }
        .dashboard-footer-logo { width: 96px; height: 36px; }
        .dashboard-footer-source { text-align: left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pio.templates["climate_analytics"] = go.layout.Template(
    layout={
        "font": {
            "family": PREMIUM_FONT_STACK,
            "color": CLIMATE_COLORS["text"],
        },
        "paper_bgcolor": CLIMATE_COLORS["background"],
        "plot_bgcolor": CLIMATE_COLORS["background"],
        "colorway": CLIMATE_SEQUENCE,
        "title": {
            "font": {
            "size": 16,
                "color": CLIMATE_COLORS["text"],
                "family": PREMIUM_FONT_STACK,
            }
        },
        "xaxis": {
            "gridcolor": "#E7ECEA",
            "linecolor": CLIMATE_COLORS["border"],
            "zerolinecolor": CLIMATE_COLORS["border"],
            "tickfont": {"color": CLIMATE_COLORS["muted"]},
            "title": {"font": {"color": CLIMATE_COLORS["muted"]}},
        },
        "yaxis": {
            "gridcolor": "#E7ECEA",
            "linecolor": CLIMATE_COLORS["border"],
            "zerolinecolor": CLIMATE_COLORS["border"],
            "tickfont": {"color": CLIMATE_COLORS["muted"]},
            "title": {"font": {"color": CLIMATE_COLORS["muted"]}},
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0)",
            "font": {"color": CLIMATE_COLORS["text"]},
        },
        "margin": {"l": 44, "r": 18, "t": 58, "b": 42},
    }
)
pio.templates.default = "climate_analytics"
px.defaults.template = "climate_analytics"
px.defaults.color_discrete_sequence = CLIMATE_SEQUENCE

st.markdown(
    f"""
    <style>
    :root {{
        --dashboard-font: Arial, Helvetica, sans-serif;
    }}

    html,
    body,
    .stApp,
    h1, h2, h3, h4, h5, h6,
    p,
    label,
    button,
    input,
    textarea,
    select,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stMetric"],
    [data-testid="stMetric"] *,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-baseweb="select"],
    [data-baseweb="input"],
    [data-baseweb="tag"] {{
        font-family: var(--dashboard-font) !important;
    }}

    .material-icons,
    .material-icons-outlined,
    .material-icons-round,
    .material-icons-sharp,
    .material-symbols-outlined,
    .material-symbols-rounded,
    .material-symbols-sharp,
    [class*="material-icons"],
    [class*="material-symbols"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: "liga" !important;
    }}

    .js-plotly-plot,
    .js-plotly-plot *,
    .plotly,
    .plotly * {{
        font-family: var(--dashboard-font) !important;
    }}

    svg text {{
        font-family: var(--dashboard-font) !important;
    }}

    .stApp {{
        background: {CLIMATE_COLORS["background"]};
        color: {CLIMATE_COLORS["text"]};
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background: radial-gradient(circle at top left, #EEF2E8 0, {CLIMATE_COLORS["background"]} 300px);
    }}

    [data-testid="stHeader"] {{
        background: rgba(244, 246, 241, 0.9);
        backdrop-filter: blur(8px);
    }}

    [data-testid="stSidebar"] {{
        background: #E8ECE1;
        border-right: 1px solid #D2D9CA;
    }}

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #1F4D3A;
        font-weight: 680;
        letter-spacing: 0;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #657369;
    }}

    [data-testid="stSidebar"] label,
    [data-testid="stWidgetLabel"] {{
        color: #232820 !important;
        -webkit-text-fill-color: #232820 !important;
        font-family: var(--dashboard-font) !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        line-height: 1.35 !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] label *,
    [data-testid="stWidgetLabel"] * {{
        color: #232820 !important;
        -webkit-text-fill-color: #232820 !important;
        font-family: var(--dashboard-font) !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        line-height: 1.35 !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: #657369;
    }}

    h1 {{
        color: {CLIMATE_COLORS["navy"]};
        font-weight: 780;
        letter-spacing: 0;
        line-height: 1.05;
        margin-bottom: 0.35rem;
    }}

    h2, h3 {{
        color: {CLIMATE_COLORS["text"]};
        letter-spacing: 0;
    }}

    div[data-testid="stCaptionContainer"] {{
        color: {CLIMATE_COLORS["muted"]};
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        width: 100%;
        max-width: 100%;
        margin: 0.8rem 0 1.35rem 0;
    }}

    .kpi-card {{
        min-width: 0;
        background: {CLIMATE_COLORS["surface"]};
        border: 1px solid {CLIMATE_COLORS["border"]};
        border-radius: 10px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 22px rgba(31, 78, 95, 0.07);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }}

    .kpi-label {{
        margin-bottom: 0.32rem;
        color: {CLIMATE_COLORS["muted"]};
        font-size: 0.82rem;
        font-weight: 580;
        line-height: 1.2;
        min-height: 2rem;
        display: flex;
        align-items: flex-start;
    }}

    .kpi-value {{
        color: {CLIMATE_COLORS["navy"]};
        font-size: clamp(1.65rem, 2.1vw, 2.45rem);
        font-weight: 650;
        line-height: 1;
        letter-spacing: 0;
        white-space: nowrap;
    }}

    .kpi-grid-compact {{
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        margin-top: 0.6rem;
    }}

    .kpi-grid-compact .kpi-value {{
        font-size: clamp(1.45rem, 1.85vw, 2.1rem);
    }}

    @media (max-width: 900px) {{
        .kpi-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}

    @media (max-width: 600px) {{
        .kpi-grid,
        .kpi-grid-compact {{
            grid-template-columns: minmax(0, 1fr);
        }}
    }}

    .selection-summary {{
        margin: 0.9rem 0 1.4rem 0;
        padding: 1rem 1.15rem 1.05rem 1.15rem;
        border: 1px solid {CLIMATE_COLORS["border"]};
        border-left: 5px solid {CLIMATE_COLORS["navy"]};
        border-radius: 10px;
        background: linear-gradient(90deg, #EDF3E9 0%, #FAFBF8 100%);
        box-shadow: 0 10px 24px rgba(31, 78, 95, 0.06);
    }}

    .selection-kicker {{
        margin-bottom: 0.62rem;
        color: {CLIMATE_COLORS["navy"]};
        font-size: 0.78rem;
        font-weight: 760;
        letter-spacing: 0.055em;
        text-transform: uppercase;
    }}

    .selection-chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.48rem;
    }}

    .selection-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        border: 1px solid #CCD7C3;
        background: #FFFFFF;
        color: {CLIMATE_COLORS["text"]};
        border-radius: 999px;
        padding: 0.38rem 0.62rem;
        font-size: 0.86rem;
        font-weight: 620;
        line-height: 1.2;
        box-shadow: 0 1px 2px rgba(31, 78, 95, 0.08);
    }}

    .selection-chip-label {{
        color: {CLIMATE_COLORS["green"]};
        font-size: 0.7rem;
        font-weight: 760;
        letter-spacing: 0.045em;
        text-transform: uppercase;
    }}

    .map-context-panel {{
        margin: 0.6rem 0 0.9rem 0;
        padding: 1rem 1.05rem;
        border: 1px solid #D7E0D5;
        border-left: 5px solid {CLIMATE_COLORS["green"]};
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(255,255,255,0.96) 0%, rgba(247,249,244,0.96) 100%);
        box-shadow: 0 12px 28px rgba(31, 78, 95, 0.06);
    }}

    .map-legend-layout {{
        display: grid;
        grid-template-columns: minmax(220px, 0.88fr) minmax(360px, 1.4fr);
        gap: 1rem;
        align-items: stretch;
    }}

    .map-context-title {{
        color: {CLIMATE_COLORS["navy"]};
        font-size: 0.96rem;
        font-weight: 780;
        line-height: 1.25;
    }}

    .map-context-subtitle {{
        margin-top: 0.28rem;
        color: {CLIMATE_COLORS["muted"]};
        font-size: 0.82rem;
        line-height: 1.35;
    }}

    .map-current-metric {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        margin-top: 0.72rem;
        padding: 0.36rem 0.56rem;
        border-radius: 999px;
        background: #EDF3E9;
        color: {CLIMATE_COLORS["navy"]};
        font-size: 0.78rem;
        font-weight: 720;
    }}

    .map-current-metric span {{
        color: {CLIMATE_COLORS["green"]};
        font-size: 0.68rem;
        font-weight: 780;
        letter-spacing: 0.045em;
        text-transform: uppercase;
    }}

    .map-legend-row {{
        display: grid;
        grid-template-columns: repeat(2, minmax(190px, 1fr));
        gap: 0.65rem;
    }}

    .map-legend-item {{
        display: flex;
        align-items: center;
        gap: 0.68rem;
        min-height: 68px;
        padding: 0.72rem 0.85rem;
        border: 1px solid #D7E0D5;
        border-radius: 10px;
        background: #FFFFFF;
        color: {CLIMATE_COLORS["text"]};
        box-shadow: 0 4px 12px rgba(31, 78, 95, 0.045);
    }}

    .map-legend-copy {{
        display: flex;
        flex-direction: column;
        gap: 0.13rem;
    }}

    .map-legend-label {{
        color: {CLIMATE_COLORS["navy"]};
        font-size: 0.78rem;
        font-weight: 760;
        line-height: 1.15;
    }}

    .map-legend-note {{
        color: {CLIMATE_COLORS["muted"]};
        font-size: 0.72rem;
        font-weight: 560;
        line-height: 1.2;
    }}

    .map-gradient-key {{
        width: 66px;
        height: 10px;
        flex: 0 0 auto;
        border-radius: 999px;
        background: linear-gradient(90deg, #E2914E 0%, #B86436 52%, #89361E 100%);
        box-shadow: inset 0 0 0 1px rgba(31, 42, 36, 0.1);
    }}

    .map-line-key {{
        width: 66px;
        height: 7px;
        flex: 0 0 auto;
        border-radius: 999px;
        background: #B86436;
        box-shadow: 0 0 0 3px rgba(255, 253, 248, 1), 0 0 0 4px rgba(31, 42, 36, 0.16);
    }}

    .map-endpoint-key,
    .map-local-key {{
        width: 13px;
        height: 13px;
        flex: 0 0 auto;
        border-radius: 50%;
        margin-left: 0.55rem;
        box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.95), 0 0 0 5px rgba(97, 116, 58, 0.22);
    }}

    .map-endpoint-key {{
        background: #1F4D3A;
    }}

    .map-local-key {{
        background: #A95B37;
        box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.95), 0 0 0 5px rgba(169, 91, 55, 0.24);
    }}

    @media (max-width: 980px) {{
        .map-legend-layout {{
            grid-template-columns: 1fr;
        }}

        .map-legend-row {{
            grid-template-columns: 1fr;
        }}
    }}

    div[data-testid="stPlotlyChart"] {{
        background: transparent;
        border: 0;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        overflow: hidden;
    }}

    div[data-testid="stPlotlyChart"] > div {{
        width: 100% !important;
        max-width: 100% !important;
    }}

    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container,
    div[data-testid="stPlotlyChart"] .svg-container {{
        width: 100% !important;
        max-width: 100% !important;
    }}

    /* Keep the chart export control compact, visible and visually consistent. */
    div[data-testid="stPlotlyChart"] .modebar {{
        top: 5px !important;
        right: 6px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 2px !important;
        background: rgba(244, 246, 241, 0.9) !important;
        border: 1px solid rgba(31, 80, 65, 0.18) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(31, 48, 43, 0.08) !important;
        opacity: 0.78 !important;
        transition: opacity 120ms ease, box-shadow 120ms ease !important;
    }}

    div[data-testid="stPlotlyChart"] .modebar:hover {{
        opacity: 1 !important;
        box-shadow: 0 3px 10px rgba(31, 48, 43, 0.14) !important;
    }}

    div[data-testid="stPlotlyChart"] .modebar-btn {{
        padding: 3px 4px !important;
    }}

    /* Replace Plotly's camera with the same compact download arrow used by tables. */
    div[data-testid="stPlotlyChart"] .modebar-btn[data-title="Download plot as a PNG"] {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 24px !important;
        min-width: 24px !important;
        height: 22px !important;
        box-sizing: border-box !important;
        opacity: 0.82 !important;
    }}

    div[data-testid="stPlotlyChart"] .modebar-btn[data-title="Download plot as a PNG"]:hover {{
        opacity: 1 !important;
    }}

    div[data-testid="stPlotlyChart"] .modebar-btn[data-title="Download plot as a PNG"] svg {{
        display: none !important;
    }}

    div[data-testid="stPlotlyChart"] .modebar-btn[data-title="Download plot as a PNG"]::before {{
        content: "";
        display: block;
        width: 16px;
        height: 16px;
        opacity: 1 !important;
        background: {CLIMATE_COLORS["navy"]};
        position: static !important;
        inset: auto !important;
        margin: 0 !important;
        transform: none !important;
        pointer-events: none !important;
        -webkit-mask: url("{DOWNLOAD_ICON_DATA_URI}") center / contain no-repeat;
        mask: url("{DOWNLOAD_ICON_DATA_URI}") center / contain no-repeat;
    }}

    div[data-testid="stPlotlyChart"] .modebar-group:empty,
    div[data-testid="stPlotlyChart"] .modebar-group:has(> .modebar-btn[data-title="Fullscreen"]),
    div[data-testid="stPlotlyChart"] .modebar-group:has(> .modebar-btn[data-title="Close fullscreen"]) {{
        display: none !important;
    }}

    div[data-testid="stPlotlyChart"] .modebar-group:has(> .modebar-btn[data-title="Download plot as a PNG"]) {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }}

    div[data-testid="stPlotlyChart"] .modebar-btn[data-title="Fullscreen"],
    div[data-testid="stPlotlyChart"] .modebar-btn[data-title="Close fullscreen"] {{
        display: none !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 0;
        border-radius: 10px;
        overflow: auto;
        box-shadow: none;
        max-width: 100%;
        box-sizing: border-box;
    }}

    /* Every analysed table keeps Streamlit's compact CSV download control. */
    div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] {{
        opacity: 1 !important;
        top: 6px !important;
        right: 6px !important;
        left: auto !important;
        z-index: 5 !important;
        width: 30px !important;
        height: 28px !important;
        padding: 0 !important;
    }}

    div[data-testid="stDataFrame"] [data-testid="stElementToolbarButtonContainer"] {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 30px !important;
        height: 28px !important;
        padding: 2px !important;
        background: rgba(244, 246, 241, 0.9) !important;
        border: 1px solid rgba(31, 80, 65, 0.18) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(31, 48, 43, 0.08) !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stDataFrame"] [data-testid="stElementToolbarButton"]:not(:has(button[aria-label="Download as CSV"])) {{
        display: none !important;
    }}

    div[data-testid="stDataFrame"] button[aria-label="Download as CSV"] {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 24px !important;
        min-width: 24px !important;
        height: 22px !important;
        padding: 3px 4px !important;
        opacity: 0.82 !important;
        color: transparent !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stDataFrame"] button[aria-label="Download as CSV"] svg {{
        display: none !important;
    }}

    div[data-testid="stDataFrame"] button[aria-label="Download as CSV"]::before {{
        content: "";
        display: block;
        width: 16px;
        height: 16px;
        background: {CLIMATE_COLORS["navy"]};
        -webkit-mask: url("{DOWNLOAD_ICON_DATA_URI}") center / contain no-repeat;
        mask: url("{DOWNLOAD_ICON_DATA_URI}") center / contain no-repeat;
    }}

    div[data-testid="stDataFrame"] button[aria-label="Download as CSV"]:hover {{
        opacity: 1 !important;
    }}

    .stButton button {{
        background: {CLIMATE_COLORS["navy"]};
        color: white;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
    }}

    .stButton button:hover {{
        background: {CLIMATE_COLORS["navy"]};
        color: white;
        border: 0;
    }}

    .st-key-data_assistant_launcher {{
        position: fixed;
        right: 1.55rem;
        bottom: 1.4rem;
        z-index: 1000000;
        width: auto;
    }}

    .st-key-data_assistant_launcher::before {{
        content: "Ask the data";
        position: absolute;
        right: 0;
        bottom: calc(100% + 0.55rem);
        padding: 0.38rem 0.58rem;
        border: 1px solid rgba(31, 77, 58, 0.12);
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.97);
        color: #1F4D3A;
        box-shadow: 0 7px 18px rgba(31, 77, 58, 0.14);
        font-size: 0.72rem;
        font-weight: 650;
        line-height: 1;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transform: translateY(4px);
        transition: opacity 140ms ease, transform 140ms ease;
    }}

    .st-key-data_assistant_launcher:hover::before {{
        opacity: 1;
        transform: translateY(0);
    }}

    .st-key-data_assistant_launcher [data-testid="stPopoverButton"] {{
        width: 50px;
        min-width: 50px;
        max-width: 50px;
        min-height: 52px;
        padding: 0;
        border: 1px solid rgba(255, 255, 255, 0.45);
        border-radius: 999px;
        background: #1F4D3A !important;
        color: #FFFFFF !important;
        box-shadow: 0 14px 34px rgba(31, 77, 58, 0.28);
        font-size: 0.92rem;
        font-weight: 680;
    }}

    .st-key-data_assistant_launcher [data-testid="stPopoverButton"] p {{
        display: none;
    }}

    .st-key-data_assistant_launcher [data-testid="stPopoverButton"] * {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }}

    .st-key-data_assistant_launcher [data-testid="stPopoverButton"]:hover {{
        background: #173E2E !important;
        border-color: rgba(255, 255, 255, 0.7);
        box-shadow: 0 16px 38px rgba(31, 77, 58, 0.34);
        transform: translateY(-1px);
    }}

    [data-testid="stPopoverBody"] {{
        width: min(540px, calc(100vw - 2rem));
        max-width: min(540px, calc(100vw - 2rem));
        max-height: min(760px, calc(100vh - 2rem));
        padding: 0.95rem;
        border: 1px solid #D7E0D5;
        border-radius: 14px;
        background: rgba(250, 251, 248, 0.98);
        box-shadow: 0 24px 60px rgba(31, 42, 36, 0.2);
        backdrop-filter: blur(14px);
        overflow-y: auto;
    }}

    .assistant-header {{
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin: -0.1rem -0.1rem 0.75rem -0.1rem;
        padding: 0.2rem 0.15rem 0.72rem 0.15rem;
        border-bottom: 1px solid #DDE4D8;
    }}

    .assistant-header .assistant-mark {{
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        flex: 0 0 38px;
        border-radius: 11px;
        background: #1F4D3A;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 1rem;
        font-weight: 760;
        box-shadow: 0 8px 18px rgba(31, 77, 58, 0.2);
    }}

    .assistant-title {{
        color: #1F2A24;
        font-size: 1rem;
        font-weight: 720;
        line-height: 1.15;
    }}

    .assistant-subtitle {{
        margin-top: 0.18rem;
        color: #657369;
        font-size: 0.76rem;
        font-weight: 520;
        line-height: 1.25;
    }}

    .assistant-dataset-card {{
        margin: 0.35rem 0 0.8rem 0;
        padding: 0.72rem 0.78rem;
        border: 1px solid #D7E0D5;
        border-radius: 10px;
        background: #FFFFFF;
        box-shadow: 0 5px 16px rgba(31, 77, 58, 0.05);
    }}

    .assistant-dataset-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
    }}

    .assistant-dataset-name {{
        color: #1F4D3A;
        font-size: 0.85rem;
        font-weight: 690;
        line-height: 1.2;
    }}

    .assistant-ready {{
        flex: 0 0 auto;
        padding: 0.22rem 0.42rem;
        border-radius: 999px;
        background: #E9F1E4;
        color: #4F672E;
        font-size: 0.65rem;
        font-weight: 760;
        letter-spacing: 0.035em;
        text-transform: uppercase;
    }}

    .assistant-dataset-meta {{
        margin-top: 0.35rem;
        color: #657369;
        font-size: 0.72rem;
        line-height: 1.35;
    }}

    .assistant-scope-note {{
        margin: 0.2rem 0 0.75rem 0;
        padding: 0.55rem 0.65rem;
        border-radius: 8px;
        background: #EDF3E9;
        color: #4E5F52;
        font-size: 0.72rem;
        line-height: 1.35;
    }}

    .assistant-scope-note strong {{
        color: #1F4D3A;
    }}

    .assistant-section-label {{
        margin: 0.65rem 0 0.38rem 0;
        color: #657369;
        font-size: 0.68rem;
        font-weight: 740;
        letter-spacing: 0.045em;
        text-transform: uppercase;
    }}

    [data-testid="stChatMessageAvatarCustom"] {{
        background: #1F4D3A !important;
        color: #FFFFFF !important;
    }}

    [data-testid="stChatMessageAvatarCustom"] * {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }}

    [class*="st-key-assistant_suggestion_"] button {{
        min-height: 42px;
        justify-content: flex-start;
        padding: 0.55rem 0.68rem;
        border: 1px solid #D7E0D5;
        border-radius: 9px;
        background: #FFFFFF;
        color: #314138;
        box-shadow: 0 3px 10px rgba(31, 77, 58, 0.045);
        font-size: 0.78rem;
        font-weight: 580;
        text-align: left;
    }}

    [class*="st-key-assistant_suggestion_"] button:hover {{
        border-color: #8DA176;
        background: #F2F6EE;
        color: #1F4D3A;
    }}

    .st-key-assistant_priority_builder {{
        margin: 0.55rem 0 0.7rem 0;
        padding: 0.75rem;
        border: 1px solid #CAD8C5;
        border-radius: 11px;
        background: linear-gradient(145deg, #F4F8F1 0%, #FFFFFF 100%);
        box-shadow: 0 7px 18px rgba(31, 77, 58, 0.07);
    }}

    .assistant-priority-title {{
        color: #1F4D3A;
        font-size: 0.88rem;
        font-weight: 720;
        line-height: 1.25;
    }}

    .assistant-priority-copy {{
        margin: 0.2rem 0 0.5rem 0;
        color: #657369;
        font-size: 0.72rem;
        line-height: 1.4;
    }}

    @media (max-width: 640px) {{
        .st-key-data_assistant_launcher {{
            right: 0.8rem;
            bottom: 0.8rem;
        }}

        .st-key-data_assistant_launcher [data-testid="stPopoverButton"] {{
            width: 48px;
            min-width: 48px;
            max-width: 48px;
            min-height: 48px;
            padding: 0;
        }}
    }}

    input[type="radio"],
    input[type="checkbox"] {{
        accent-color: {CLIMATE_COLORS["navy"]};
    }}

    label[data-baseweb="radio"]:has(input[type="radio"]:checked)
    > div:first-child {{
        background-color: {CLIMATE_COLORS["navy"]} !important;
        border-color: {CLIMATE_COLORS["navy"]} !important;
    }}

    div[data-baseweb="tag"],
    span[data-baseweb="tag"],
    [data-testid="stMultiSelect"] div[data-baseweb="tag"] {{
        background-color: #1F4D3A !important;
        border-color: #1F4D3A !important;
        color: #FFFFFF !important;
    }}

    div[data-baseweb="tag"] span,
    span[data-baseweb="tag"] span,
    [data-testid="stMultiSelect"] div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] svg,
    span[data-baseweb="tag"] svg,
    div[data-baseweb="tag"] path {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }}

    [data-testid="stSlider"] [role="slider"] {{
        background-color: #1F4D3A !important;
        border-color: #1F4D3A !important;
        box-shadow: 0 0 0 1px #1F4D3A !important;
    }}

    /* Keep every slider consistent across all dashboard pages. */
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div,
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div,
    [data-testid="stSlider"] [data-baseweb="slider"]
        div[style*="rgb(255, 75, 75)"],
    [data-testid="stSlider"] [data-baseweb="slider"]
        div[style*="#ff4b4b"],
    [data-testid="stSlider"] [data-baseweb="slider"]
        div[style*="#FF4B4B"] {{
        background-color: #1F4D3A !important;
        background-image: none !important;
    }}

    [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSlider"] output {{
        color: #1F4D3A !important;
    }}

    /* Scenario sliders show the selected value above the control, so the
       duplicated 0/100 endpoint labels add clutter and are hidden here only. */
    [class*="st-key-scenario_ship_reduction_"] [data-testid="stTickBarMin"],
    [class*="st-key-scenario_ship_reduction_"] [data-testid="stTickBarMax"],
    [class*="st-key-scenario_ship_reduction_"] [data-testid="stSliderTickBar"] {{
        display: none !important;
    }}

    /* Sidebar sliders retain the selected, minimum and maximum values. Keep all
       three as plain text in the same colour, without coloured label boxes. */
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 100% !important;
        height: 1.35rem !important;
        min-height: 1.35rem !important;
        margin-top: 0.2rem !important;
        align-items: flex-start !important;
        justify-content: space-between !important;
        overflow: visible !important;
        background: transparent !important;
        background-color: transparent !important;
    }}

    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] > div,
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] *,
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"],
    [data-testid="stSidebar"] [data-testid="stTickBarMin"] *,
    [data-testid="stSidebar"] [data-testid="stTickBarMax"] * {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #1F4D3A !important;
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: 0 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] p {{
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.15 !important;
        font-size: 0.88rem !important;
        font-weight: 400 !important;
    }}

    .stApp [data-testid="stSidebar"] [data-testid="stSlider"]
        [data-testid="stSliderTickBar"] [data-testid="stMarkdownContainer"] {{
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        border: 0 !important;
        border-radius: 0 !important;
        color: #1F4D3A !important;
    }}

    /* Main-page sliders use the same plain endpoint-label treatment. */
    .stApp [data-testid="stMain"] [data-testid="stSliderTickBar"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 100% !important;
        min-height: 1.35rem !important;
        justify-content: space-between !important;
        background: transparent !important;
        background-color: transparent !important;
        overflow: visible !important;
    }}

    .stApp [data-testid="stMain"] [data-testid="stSlider"]
        [data-testid="stSliderTickBar"] [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMain"] [data-testid="stSelectSlider"]
        [data-testid="stSliderTickBar"] [data-testid="stMarkdownContainer"] {{
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        border: 0 !important;
        border-radius: 0 !important;
        color: #1F4D3A !important;
    }}

    .stApp [data-testid="stMain"] [data-testid="stSliderTickBar"] p {{
        color: #1F4D3A !important;
        background: transparent !important;
        background-color: transparent !important;
    }}

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        min-height: 44px;
        border-radius: 8px;
        border-color: #D7E0D5;
        background-color: #FFFFFF;
    }}

    [data-baseweb="select"] *,
    [data-baseweb="input"] input,
    [data-baseweb="popover"] * {{
        color: {CLIMATE_COLORS["text"]};
        font-family: var(--dashboard-font) !important;
        font-size: 15px !important;
        line-height: 1.35 !important;
    }}

    /* BaseWeb renders open menus in a portal outside the sidebar. Explicit
       colours prevent browser or operating-system dark-mode preferences from
       producing dark-on-dark dropdown text. */
    [data-baseweb="popover"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {{
        background-color: #FFFFFF !important;
        color: #1F2A24 !important;
    }}

    [data-baseweb="popover"] [role="option"] {{
        min-height: 42px !important;
        padding: 10px 12px !important;
        background-color: #FFFFFF !important;
        color: #1F2A24 !important;
        font-family: var(--dashboard-font) !important;
        font-size: 15px !important;
        line-height: 1.35 !important;
    }}

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [role="option"][aria-selected="true"] {{
        background-color: #E8ECE1 !important;
        color: #1F4D3A !important;
    }}

    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"] > div:focus-within {{
        border-color: {CLIMATE_COLORS["teal"]} !important;
        box-shadow: 0 0 0 1px {CLIMATE_COLORS["teal"]} !important;
    }}

    /* Streamlit renders widget-help text in a document-level portal. Keep
       every native '?' explanation as compact as the route-search help card. */
    [data-testid="stTooltipContent"],
    [data-testid="stTooltipErrorContent"] {{
        width: max-content !important;
        min-width: 220px !important;
        max-width: min(360px, calc(100vw - 32px)) !important;
        padding: 12px 16px !important;
        border: 1px solid #D7E0D5 !important;
        border-radius: 10px !important;
        background: #F7F8F5 !important;
        color: #1F2A24 !important;
        box-shadow: 0 4px 14px rgba(31, 42, 36, 0.12) !important;
        font-family: var(--dashboard-font) !important;
        font-size: 13px !important;
        line-height: 1.45 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }}

    [data-testid="stTooltipContent"] p,
    [data-testid="stTooltipErrorContent"] p {{
        margin: 0 !important;
        color: #1F2A24 !important;
        font-family: var(--dashboard-font) !important;
        font-size: 13px !important;
        line-height: 1.45 !important;
    }}

    @media (max-width: 1180px) {{
        [data-testid="column"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }}

        div[data-testid="stPlotlyChart"] {{
            padding: 0.25rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# 4. Load and validate the enriched dataset

@st.cache_data(show_spinner="Loading enriched parquet file...")
def load_data(parquet_path: str) -> pd.DataFrame:
    """Load the enriched voyage-level dataset."""
    path = Path(parquet_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    available_columns = pq.read_schema(path).names
    columns = [col for col in COLUMNS_TO_LOAD if col in available_columns]

    df = pd.read_parquet(path, columns=columns)

    if "departure_time" in df.columns:
        df["departure_time"] = pd.to_datetime(df["departure_time"], errors="coerce")

    if "arrival_time" in df.columns:
        df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")

    if "departure_year" not in df.columns:
        df["departure_year"] = df["departure_time"].dt.year

    # Apply the study period in memory. The source Parquet remains unchanged.
    df = df[
        df["departure_year"].between(
            ANALYSIS_START_YEAR,
            ANALYSIS_END_YEAR,
            inclusive="both",
        )
    ].copy()

    if "departure_month" not in df.columns:
        df["departure_month"] = df["departure_time"].dt.to_period("M").astype(str)

    df["country_o"] = df["country_o"].fillna("Unknown")
    df["country_d"] = df["country_d"].fillna("Unknown")

    if "ship_type_clean" not in df.columns:
        df["ship_type_clean"] = "Unknown"
    else:
        df["ship_type_clean"] = df["ship_type_clean"].fillna("Unknown")

    if "voyage_scope" not in df.columns:
        df["voyage_scope"] = np.where(
            df["country_o"].eq(df["country_d"]),
            "Domestic",
            "International",
        )

    if "route_directional" not in df.columns:
        df["route_directional"] = (
            df["country_o"].astype(str)
            + " → "
            + df["country_d"].astype(str)
        )

    if "country_pair" not in df.columns:
        pair_left = df[["country_o", "country_d"]].astype(str).min(axis=1)
        pair_right = df[["country_o", "country_d"]].astype(str).max(axis=1)
        df["country_pair"] = pair_left + " ↔ " + pair_right

    if "origin_port_display" not in df.columns:
        df["origin_port_display"] = df.get("port_id_o", "Unknown").astype(str)

    if "destination_port_display" not in df.columns:
        df["destination_port_display"] = df.get("port_id_d", "Unknown").astype(str)

    if "port_route_display" not in df.columns:
        df["port_route_display"] = (
            df["origin_port_display"].astype(str)
            + " → "
            + df["destination_port_display"].astype(str)
        )

    if "CO2_per_km" not in df.columns:
        df["CO2_per_km"] = np.where(
            df["delta_dist_km"] > 0,
            df["CO2"] / df["delta_dist_km"],
            np.nan,
        )

    if "CO2_per_dwt_km" not in df.columns:
        df["CO2_per_dwt_km"] = np.where(
            (df["delta_dist_km"] > 0) & (df["dwt"] > 0),
            df["CO2"] / (df["delta_dist_km"] * df["dwt"]),
            np.nan,
        )

    if "vessel_age_at_departure" in df.columns:
        age_labels = [
            "0–5 years",
            "6–10 years",
            "11–15 years",
            "16–20 years",
            "21–25 years",
            "26–30 years",
            "31+ years",
        ]
        vessel_age = pd.to_numeric(df["vessel_age_at_departure"], errors="coerce")
        valid_vessel_age = vessel_age.where(vessel_age >= 0)
        age_group = pd.cut(
            valid_vessel_age,
            bins=[-0.001, 5, 10, 15, 20, 25, 30, np.inf],
            labels=age_labels,
            ordered=True,
            include_lowest=True,
        )
        age_group = age_group.cat.add_categories(["Unknown / invalid age"])
        df["vessel_age_group"] = age_group.fillna("Unknown / invalid age")

    return df


# 5. Helper functions

KG_PER_MILLION_TONNES = 1_000_000_000


def kg_to_million_tonnes(value: float) -> float:
    """Convert kilograms to the shipping-standard unit of million tonnes."""
    return value / KG_PER_MILLION_TONNES


def format_million_tonnes(value: float) -> str:
    """Format million tonnes without unreadable raw kilogram values."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"

def format_number(value: float) -> str:
    """Format large values for KPI cards."""
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def format_co2_label(text: object) -> str:
    """Format user-facing CO2 labels with a subscript 2."""
    return str(text).replace("CO2", "CO₂").replace("co2", "CO₂")


def display_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Display dataframe columns with presentation-friendly labels and units."""
    display_df = df.copy()
    absolute_co2_columns = {
        "CO2",
        "total_CO2",
        "total_co2_kg",
        "ship_co2_kg",
        "corridor_co2_kg",
        "start_co2_kg",
        "end_co2_kg",
        "absolute_change_kg",
    }
    for column in absolute_co2_columns.intersection(display_df.columns):
        display_df[column] = display_df[column] / KG_PER_MILLION_TONNES

    display_labels = {
        "ship_type_clean": "Ship type",
        "CO2": "CO₂ (million tonnes)",
        "CO2_million_tonnes": "CO₂ (million tonnes)",
        "total_CO2": "Total CO₂ (million tonnes)",
        "total_co2_kg": "Total CO₂ (million tonnes)",
        "ship_co2_kg": "Ship CO₂ (million tonnes)",
        "corridor_co2_kg": "Corridor CO₂ (million tonnes)",
        "start_co2_kg": "Start CO₂ (million tonnes)",
        "end_co2_kg": "End CO₂ (million tonnes)",
        "absolute_change_kg": "Absolute change (million tonnes)",
        "mean_CO2": "Mean CO₂ per voyage (kg)",
        "mean_distance_km": "Mean distance (km)",
        "mean_CO2_per_km": "Mean CO₂ intensity (kg/km)",
        "weighted_CO2_per_km": "Distance-weighted CO₂ intensity (kg/km)",
        "weighted_CO2_per_dwt_km": "Weighted CO₂ intensity (kg/DWT-km)",
        "mean_dwt": "Mean deadweight tonnage (DWT)",
        "mean_engine_power": "Mean engine power (kW)",
        "vessel_age_group": "Vessel age group",
        "unique_vessels": "Unique vessels (IMO)",
        "voyage_count": "Voyages (records)",
    }
    renamed_columns = {
        col: display_labels.get(col, format_co2_label(col)) for col in display_df.columns
    }
    display_df = display_df.rename(
        columns=renamed_columns
    )
    integer_columns = {
        "voyage_count",
        "unique_vessels",
        "vessel_count",
        "vessels",
        "departure_year",
        "imo",
        "mmsi",
    }
    default_column_config = {}
    for original_column, display_column in renamed_columns.items():
        if not pd.api.types.is_numeric_dtype(df[original_column]):
            continue
        if "kg/DWT-km" in str(display_column):
            number_format = "%.6f"
        elif original_column in integer_columns or original_column.endswith("_count"):
            number_format = "%.0f"
        else:
            number_format = "%.2f"
        default_column_config[display_column] = st.column_config.NumberColumn(
            format=number_format
        )
    supplied_column_config = kwargs.pop("column_config", {}) or {}
    kwargs["column_config"] = {**default_column_config, **supplied_column_config}
    st.dataframe(display_df, **kwargs)


def format_metric_label(metric_name: str) -> str:
    """Format metric names for widgets while keeping internal values unchanged."""
    if metric_name == "Total CO2":
        return "Total CO₂ (million tonnes)"
    if metric_name == "CO2 per km":
        return "CO₂ intensity (kg/km)"
    if metric_name == "CO2 per DWT-km":
        return "CO₂ intensity (kg/DWT-km)"
    return format_co2_label(metric_name)


def format_metric_value(metric_name: str, value: float) -> str:
    """Format selected metric values with suitable precision."""
    if pd.isna(value):
        return "N/A"
    if metric_name == "CO2 per DWT-km":
        return f"{value:.6f}"
    if metric_name == "CO2 per km":
        return f"{value:,.2f}"
    if metric_name == "Voyage count":
        return f"{value:,.0f}"
    return format_million_tonnes(value)


def metric_column(metric_name: str) -> str:
    """Map dashboard metric names to dataframe columns."""
    metric_map = {
        "Total CO2": "CO2_million_tonnes",
        "CO2 per km": "CO2_per_km",
        "CO2 per DWT-km": "CO2_per_dwt_km",
        "Voyage count": "voyage_count",
    }
    return metric_map[metric_name]


def aggregate_metric(
    df: pd.DataFrame,
    group_col: str,
    metric_name: str,
    top_n: int,
    min_voyages: int = 0,
) -> pd.DataFrame:
    """Aggregate data for ranked bar charts."""
    metric_col = metric_column(metric_name)

    if metric_name == "Voyage count":
        out = (
            df.groupby(group_col, dropna=False)
            .size()
            .reset_index(name="voyage_count")
            .sort_values("voyage_count", ascending=False)
        )
    elif metric_name == "Total CO2":
        out = (
            df.groupby(group_col, dropna=False)["CO2"]
            .sum()
            .reset_index()
            .sort_values("CO2", ascending=False)
            .rename(columns={"CO2": "CO2_million_tonnes"})
        )
        out["CO2_million_tonnes"] = out["CO2_million_tonnes"] / KG_PER_MILLION_TONNES
    elif metric_name == "CO2 per km":
        valid = df[
            df["CO2"].notna()
            & df["delta_dist_km"].notna()
            & (df["delta_dist_km"] > 0)
        ]
        out = (
            valid.groupby(group_col, dropna=False)
            .agg(
                voyage_count=("CO2", "size"),
                total_CO2=("CO2", "sum"),
                total_distance_km=("delta_dist_km", "sum"),
            )
            .reset_index()
        )
        out[metric_col] = out["total_CO2"] / out["total_distance_km"]
        out = out.sort_values(metric_col, ascending=False)
    elif metric_name == "CO2 per DWT-km":
        valid = df[
            df["CO2"].notna()
            & df["delta_dist_km"].notna()
            & (df["delta_dist_km"] > 0)
            & df["dwt"].notna()
            & (df["dwt"] > 0)
        ].copy()
        valid["dwt_km"] = valid["delta_dist_km"] * valid["dwt"]
        out = (
            valid.groupby(group_col, dropna=False)
            .agg(
                voyage_count=("CO2", "size"),
                total_CO2=("CO2", "sum"),
                total_dwt_km=("dwt_km", "sum"),
            )
            .reset_index()
        )
        out[metric_col] = out["total_CO2"] / out["total_dwt_km"]
        out = out.sort_values(metric_col, ascending=False)
    else:
        out = (
            df.groupby(group_col, dropna=False)
            .agg(
                voyage_count=("CO2", "size"),
                value=(metric_col, "mean"),
            )
            .reset_index()
            .rename(columns={"value": metric_col})
            .sort_values(metric_col, ascending=False)
        )

    if min_voyages > 0 and "voyage_count" in out.columns:
        out = out[out["voyage_count"] >= min_voyages]
    return out.head(top_n)


def ranked_filter_options(df: pd.DataFrame, group_col: str, limit: int | None = None) -> list[str]:
    """Return filter options ranked by total CO2."""
    if group_col not in df.columns or df.empty:
        return []

    ranked = (
        df.groupby(group_col, dropna=False)["CO2"]
        .sum()
        .reset_index()
        .sort_values("CO2", ascending=False)
    )

    options = ranked[group_col].dropna().astype(str).tolist()
    if limit is not None:
        options = options[:limit]
    return options


def country_name(code: str) -> str:
    """Return a readable country name for an ISO3 code when available."""
    clean_code = str(code).strip().upper()
    return COUNTRY_NAME_MAP.get(clean_code, clean_code)


def format_country_pair_option(option: str) -> str:
    """Display country-pair route options with both ISO3 codes and country names."""
    if option == "All":
        return option

    parts = [part.strip() for part in str(option).split("↔")]
    if len(parts) != 2:
        return str(option)

    left, right = parts
    return f"{left} ↔ {right} ({country_name(left)} ↔ {country_name(right)})"


def format_kpler_vessel_type(option: str) -> str:
    """Present Kpler's uppercase categories consistently without changing values."""
    text = str(option).strip()
    if text == "All":
        return text

    formatted = text.title()
    for acronym in ("LPG", "LNG", "FSO", "FPSO", "VLCC", "ULCC"):
        formatted = re.sub(
            rf"\b{re.escape(acronym.title())}\b",
            acronym,
            formatted,
        )
    return formatted


def filter_options_by_display(
    options: list[str],
    query: str,
    display_func=str,
    limit: int | None = None,
) -> list[str]:
    """Filter dropdown options using what the user sees in the UI."""
    terms = [term for term in query.lower().split() if term]

    if terms:
        options = [
            option
            for option in options
            if all(term in display_func(option).lower() for term in terms)
        ]

    if limit is not None:
        options = options[:limit]

    return options


def searchable_route_select(
    label: str,
    raw_options: list[str],
    *,
    key: str,
    placeholder: str,
    display_func=str,
    default_limit: int = 50,
    search_limit: int = 250,
    help_text: str | None = None,
) -> str:
    """Render a searchable route selector with All as the default option."""
    display_to_raw = {
        display_func(option): option
        for option in raw_options
    }

    def search_options(search_term: str) -> list[str]:
        limit = search_limit if search_term.strip() else default_limit
        matched = filter_options_by_display(
            raw_options,
            search_term,
            display_func=display_func,
            limit=limit,
        )
        return ["All"] + [display_func(option) for option in matched]

    default_options = ["All"] + [
        display_func(option)
        for option in raw_options[:default_limit]
    ]

    if st_searchbox is not None:
        selected_display = st_searchbox(
            search_options,
            placeholder=placeholder,
            label=label,
            default="All",
            default_options=default_options,
            clear_on_submit=False,
            edit_after_submit="option",
            style_overrides={
                "wrapper": {
                    "backgroundColor": "#E8ECE1",
                    "color": "#232820",
                    "WebkitTextFillColor": "#232820",
                    "fontFamily": PREMIUM_FONT_STACK,
                    "fontSize": "14px",
                    "fontWeight": "400",
                    "lineHeight": "1.35",
                    "opacity": "1",
                },
                "searchbox": {
                    "control": {
                        "minHeight": "44px",
                        "backgroundColor": "#FFFFFF",
                        "border": "1px solid #D7E0D5",
                        "borderRadius": "8px",
                        "boxShadow": "none",
                        "&:hover": {
                            "border": "1px solid #B8C5B5",
                        },
                    },
                    "placeholder": {
                        "color": "#7A837D",
                        "fontSize": "15px",
                        "lineHeight": "1.35",
                    },
                    "singleValue": {
                        "color": "#1F2A24",
                        "fontSize": "15px",
                        "lineHeight": "1.35",
                    },
                    "input": {
                        "color": "#1F2A24",
                        "fontSize": "15px",
                        "lineHeight": "1.35",
                    },
                    "menuList": {
                        "backgroundColor": "#FFFFFF",
                        "color": "#1F2A24",
                        "fontFamily": PREMIUM_FONT_STACK,
                        "fontSize": "15px",
                        "lineHeight": "1.35",
                        "paddingTop": "4px",
                        "paddingBottom": "4px",
                    },
                    "option": {
                        "color": "#1F2A24",
                        "backgroundColor": "#FFFFFF",
                        "highlightColor": "#E8ECE1",
                    },
                },
            },
            key=key,
            help=help_text,
        )
    else:
        selected_display = st.sidebar.selectbox(
            label,
            default_options,
            index=0,
            help=help_text,
        )

    if not selected_display or selected_display == "All":
        return "All"

    return display_to_raw.get(str(selected_display), str(selected_display))


def page_intent(page: str) -> str:
    """Return the user task implied by each dashboard view."""
    intent_map = {
        "Overview": "Understand scale and context",
        "Route Explorer": "Identify emission corridors",
        "Ship and Vessel Analysis": "Compare vessel contribution",
        "Port Analysis": "Locate port hotspots",
        "Emission Intensity": "Find high-intensity operations",
        "Animated Route Map": "Trace spatial-temporal patterns",
    }
    return intent_map.get(page, "Explore emissions")


def render_selection_summary(
    active_filters: list[str],
    page: str,
    metric_name: str,
    top_n: int,
    route_mode: str = "Country pair",
) -> None:
    """Render the current analytical focus as compact visual chips."""
    chips = [
        ("View", page),
        ("Task", page_intent(page)),
    ]

    if active_filters:
        chips.extend(("Scope", active_filter) for active_filter in active_filters)
    else:
        chips.append(("Scope", "Full dataset"))

    if page == "Route Explorer":
        chips.extend(
            [
                ("Analysis level", route_mode),
                ("Ranking metric", format_metric_label(metric_name)),
                ("Routes shown", str(top_n)),
            ]
        )
    elif page == "Ship and Vessel Analysis":
        chips.extend(
            [
                ("Comparison metric", format_metric_label(metric_name)),
                ("Ship types shown", str(top_n)),
            ]
        )
    elif page == "Port Analysis":
        chips.append(("Ports shown", str(top_n)))
    elif page == "Animated Route Map":
        chips.extend(
            [
                ("Map metric", format_metric_label(metric_name)),
                ("Routes shown", str(top_n)),
            ]
        )

    chip_html = "".join(
        (
            '<span class="selection-chip">'
            f'<span class="selection-chip-label">{html.escape(label)}</span>'
            f'{html.escape(value)}'
            "</span>"
        )
        for label, value in chips
    )

    st.markdown(
        f"""
        <div class="selection-summary">
            <div class="selection-kicker">Current analytical focus</div>
            <div class="selection-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_pie_chart(fig: go.Figure) -> go.Figure:
    """Make pie chart labels readable on dark emission colours."""
    fig.update_traces(
        textfont={
            "color": "#FFFFFF",
            "size": 14,
            "family": PREMIUM_FONT_STACK,
        },
        insidetextfont={
            "color": "#FFFFFF",
            "size": 14,
            "family": PREMIUM_FONT_STACK,
        },
        marker={
            "line": {
                "color": "#FFFFFF",
                "width": 1.5,
            }
        },
    )
    return fig


def wrap_chart_title(title: str, line_length: int = 34) -> str:
    """Wrap long Plotly titles so they fit inside responsive chart cards."""
    if not title:
        return title

    return "<br>".join(
        textwrap.wrap(
            title,
            width=line_length,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


PLOTLY_MAP_TRACE_TYPES = {
    "choropleth",
    "choroplethmap",
    "choroplethmapbox",
    "densitymap",
    "densitymapbox",
    "scattergeo",
    "scattermap",
    "scattermapbox",
}


def is_plotly_map(fig: go.Figure) -> bool:
    """Return True when a figure contains a geographic map trace."""
    return any(str(getattr(trace, "type", "")).lower() in PLOTLY_MAP_TRACE_TYPES for trace in fig.data)


def chart_download_filename(title: object) -> str:
    """Build a short, filesystem-safe filename for the exported chart image."""
    plain_title = re.sub(r"<[^>]+>", " ", str(title or "analysis_chart"))
    slug = re.sub(r"[^a-z0-9]+", "_", plain_title.lower()).strip("_")
    return f"north_sea_emissions_{slug[:72] or 'analysis_chart'}"


def add_ship_type_colour_key(fig: go.Figure, named_colour: str) -> None:
    """Add a compact in-chart key that remains visible in downloaded PNG files."""
    fig.update_layout(showlegend=False)
    fig.add_annotation(
        x=0,
        y=1.01,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        align="left",
        showarrow=False,
        text=(
            f"<span style='color:{named_colour}'>■</span> Named ship type"
            f"<br><span style='color:{CLIMATE_COLORS['muted']}'>■</span> "
            "Other / unknown (combined)"
        ),
        font={"size": 10, "color": CLIMATE_COLORS["muted"]},
    )


def render_plotly_chart(
    fig: go.Figure,
    height: int = 420,
    extra_top_margin: int = 0,
    extra_bottom_margin: int = 0,
) -> None:
    """Render a responsive chart with image export enabled for non-map figures."""
    title = fig.layout.title.text
    wrapped_title = wrap_chart_title(str(title)) if title else ""
    title_lines = wrapped_title.count("<br>") + 1 if wrapped_title else 0
    top_margin = 48 + max(title_lines - 1, 0) * 18 + extra_top_margin

    fig.update_layout(
        autosize=True,
        height=height,
        title={
            "text": wrapped_title,
            "x": 0.01,
            "xanchor": "left",
            "font": {
                "size": 15,
                "family": PREMIUM_FONT_STACK,
                "color": CLIMATE_COLORS["text"],
            },
        },
        margin={"l": 44, "r": 16, "t": top_margin, "b": 42 + extra_bottom_margin},
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        paper_bgcolor=CLIMATE_COLORS["background"],
        plot_bgcolor=CLIMATE_COLORS["background"],
        dragmode=False,
    )
    fig.update_xaxes(automargin=True, title_standoff=8)
    fig.update_yaxes(automargin=True, title_standoff=8)

    x_axis_title = str(fig.layout.xaxis.title.text or "")
    y_axis_title = str(fig.layout.yaxis.title.text or "")
    if "million tonnes" in x_axis_title:
        fig.update_xaxes(tickformat=".2f")
        fig.update_traces(xhoverformat=".2f")
    if "million tonnes" in y_axis_title:
        fig.update_yaxes(tickformat=".2f")
        fig.update_traces(yhoverformat=".2f")

    allow_image_download = not is_plotly_map(fig)
    chart_config = {
        "responsive": True,
        "displayModeBar": allow_image_download,
        "displaylogo": False,
        "scrollZoom": False,
        "doubleClick": "reset",
    }
    if allow_image_download:
        chart_config.update(
            {
                "modeBarButtonsToRemove": [
                    "zoom2d",
                    "pan2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                    "toggleSpikelines",
                ],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": chart_download_filename(title),
                    "scale": 2,
                },
            }
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=chart_config,
    )


def render_animated_europe_globe(fig: go.Figure, height: int = 590) -> None:
    """Render the Europe globe with an automatic, accessibility-aware loop."""
    frame_names = [frame.name for frame in fig.frames]
    post_script = f"""
        (function () {{
            const chart = document.getElementById('{{plot_id}}');
            const frames = {json.dumps(frame_names)};
            const reduceMotion = window.matchMedia(
                '(prefers-reduced-motion: reduce)'
            ).matches;

            function playLoop() {{
                if (reduceMotion || !frames.length || !document.body.contains(chart)) return;
                Plotly.animate(chart, frames, {{
                    mode: 'immediate',
                    frame: {{duration: 150, redraw: true}},
                    transition: {{duration: 120, easing: 'sine-in-out'}}
                }}).then(function () {{
                    window.setTimeout(playLoop, 180);
                }});
            }}

            window.setTimeout(playLoop, 350);
        }})();
    """
    chart_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displayModeBar": False,
            "displaylogo": False,
            "scrollZoom": False,
        },
        auto_play=False,
        default_width="100%",
        default_height=f"{height}px",
        post_script=post_script,
    )
    components.html(chart_html, height=height, scrolling=False)


def require_pydeck() -> None:
    """Stop the app with a clear message if PyDeck is not installed."""
    if pdk is None:
        st.error("PyDeck is not installed. Please run: pip install pydeck")
        st.stop()


MAP_METRIC_CONFIG = {
    "Total CO2": {
        "column": "total_CO2",
        "label": "Total CO₂",
        "unit": "million tonnes",
        "low_color": [226, 145, 78, 215],
        "high_color": [137, 54, 30, 248],
    },
    "CO2 per km": {
        "column": "weighted_CO2_per_km",
        "label": "Weighted CO₂ per km",
        "unit": "kg/km",
        "low_color": [224, 174, 69, 215],
        "high_color": [118, 82, 18, 248],
    },
    "CO2 per DWT-km": {
        "column": "weighted_CO2_per_dwt_km",
        "label": "Weighted CO₂ per DWT-km",
        "unit": "kg/DWT-km",
        "low_color": [74, 163, 145, 215],
        "high_color": [18, 82, 67, 248],
    },
    "Voyage count": {
        "column": "voyage_count",
        "label": "Voyage count",
        "unit": "voyages",
        "low_color": [112, 153, 83, 215],
        "high_color": [38, 73, 31, 248],
    },
}


def get_map_metric_config(metric_name: str) -> dict:
    """Return map styling and aggregation settings for the selected sidebar metric."""
    return MAP_METRIC_CONFIG.get(metric_name, MAP_METRIC_CONFIG["Total CO2"])


def interpolate_color(value: float, max_value: float, low_color: list[int], high_color: list[int]) -> list[int]:
    """Interpolate between two RGBA colours using the metric value."""
    if pd.isna(value) or pd.isna(max_value) or max_value <= 0:
        return [120, 120, 120, 120]

    ratio = min(max(value / max_value, 0), 1)
    return [
        int(low + (high - low) * ratio)
        for low, high in zip(low_color, high_color)
    ]


def approximate_route_distance_km(route_df: pd.DataFrame) -> pd.Series:
    """Approximate source-target distance for filtering unstable map loops."""
    lat1 = np.radians(route_df["origin_lat"].astype(float))
    lat2 = np.radians(route_df["destination_lat"].astype(float))
    dlat = lat2 - lat1
    dlon = np.radians(route_df["destination_lon"].astype(float) - route_df["origin_lon"].astype(float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))


def apply_map_metric_style(route_summary: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Sort, label, colour, and size routes using the selected sidebar metric."""
    metric_config = get_map_metric_config(metric_name)
    metric_col = metric_config["column"]

    route_summary = (
        route_summary
        .dropna(subset=[metric_col])
        .sort_values(metric_col, ascending=False)
        .copy()
    )

    if route_summary.empty:
        return route_summary

    max_value = route_summary[metric_col].max()
    route_summary["map_metric_name"] = metric_config["label"]
    route_summary["map_metric_unit"] = metric_config["unit"]
    route_summary["map_metric_value"] = route_summary[metric_col]
    if metric_name == "Total CO2":
        route_summary["map_metric_label"] = route_summary["map_metric_value"].apply(
            lambda value: format_million_tonnes(kg_to_million_tonnes(value))
        )
    else:
        route_summary["map_metric_label"] = route_summary["map_metric_value"].apply(
            lambda value: "N/A" if pd.isna(value) else f"{value:,.2f}"
        )
    route_summary["line_width"] = (
        route_summary["map_metric_value"] / max_value * 9
    ).clip(lower=1.8, upper=9)
    route_summary["shadow_width"] = (route_summary["line_width"] + 3.2).clip(upper=13)
    route_summary["route_color"] = route_summary["map_metric_value"].apply(
        lambda value: interpolate_color(
            value,
            max_value,
            metric_config["low_color"],
            metric_config["high_color"],
        )
    )

    return route_summary


def get_initial_view(route_df: pd.DataFrame) -> pdk.ViewState:
    """Set the initial map viewport from available route coordinates."""
    lat_values = pd.concat([route_df["origin_lat"], route_df["destination_lat"]]).dropna()
    lon_values = pd.concat([route_df["origin_lon"], route_df["destination_lon"]]).dropna()

    if lat_values.empty or lon_values.empty:
        return pdk.ViewState(latitude=56, longitude=4, zoom=2.4, pitch=35, bearing=0)

    lat_range = max(float(lat_values.max() - lat_values.min()), 0.1)
    lon_range = max(float(lon_values.max() - lon_values.min()), 0.1)
    max_range = max(lat_range, lon_range)

    if max_range < 2:
        zoom = 5.2
    elif max_range < 5:
        zoom = 4.2
    elif max_range < 10:
        zoom = 3.4
    elif max_range < 20:
        zoom = 2.6
    else:
        zoom = 1.8

    return pdk.ViewState(
        latitude=float(lat_values.mean()),
        longitude=float(lon_values.mean()),
        zoom=zoom,
        pitch=38,
        bearing=0,
    )


@st.cache_data(show_spinner=False)
def build_temporal_route_summary(
    df: pd.DataFrame,
    period: str,
    top_n: int,
    cumulative: bool,
    metric_name: str,
) -> pd.DataFrame:
    """Aggregate port-to-port routes for a selected month."""
    valid = df.dropna(
        subset=[
            "origin_lat",
            "origin_lon",
            "destination_lat",
            "destination_lon",
            "departure_month",
        ]
    ).copy()

    if cumulative:
        valid = valid[valid["departure_month"] <= period]
    else:
        valid = valid[valid["departure_month"] == period]

    dwt_valid = (
        valid["CO2"].notna()
        & valid["delta_dist_km"].notna()
        & (valid["delta_dist_km"] > 0)
        & valid["dwt"].notna()
        & (valid["dwt"] > 0)
    )
    valid["dwt_valid_CO2"] = valid["CO2"].where(dwt_valid)
    valid["total_dwt_km"] = (
        valid["delta_dist_km"] * valid["dwt"]
    ).where(dwt_valid)

    route_summary = (
        valid.groupby(
            [
                "port_route_display",
                "origin_port_display",
                "destination_port_display",
                "origin_lat",
                "origin_lon",
                "destination_lat",
                "destination_lon",
            ],
            dropna=False,
        )
        .agg(
            total_CO2=("CO2", "sum"),
            total_distance_km=("delta_dist_km", "sum"),
            voyage_count=("CO2", "size"),
            dwt_valid_CO2=("dwt_valid_CO2", "sum"),
            total_dwt_km=("total_dwt_km", "sum"),
        )
        .reset_index()
    )
    route_summary["weighted_CO2_per_km"] = (
        route_summary["total_CO2"]
        / route_summary["total_distance_km"].replace(0, np.nan)
    )
    route_summary["weighted_CO2_per_dwt_km"] = (
        route_summary["dwt_valid_CO2"]
        / route_summary["total_dwt_km"].replace(0, np.nan)
    )

    styled_routes = apply_map_metric_style(route_summary, metric_name)
    # Keep cached map aggregates aligned with the current visual specification.
    styled_routes["map_style_revision"] = "high-contrast-v2"
    return styled_routes.head(top_n)


def render_map_context_panel(metric_name: str, route_count: int, arc_count: int) -> None:
    """Render a compact legend that explains how to read the route map."""
    local_count = max(route_count - arc_count, 0)
    st.markdown(
        f"""
        <div class="map-context-panel">
            <div class="map-legend-layout">
                <div class="map-context-copy">
                    <div class="map-context-title">Route and port activity map</div>
                    <div class="map-context-subtitle">
                        Lines show visible port-to-port flows; fixed port markers retain endpoints and local short movements.
                    </div>
                    <div class="map-current-metric"><span>Current metric</span>{html.escape(metric_name)}</div>
                    <div class="map-context-subtitle">
                        {arc_count:,} route arcs shown · {local_count:,} very short or same-port route records shown as port activity.
                    </div>
                </div>
                <div class="map-legend-row">
                    <div class="map-legend-item">
                        <span class="map-line-key"></span>
                        <div class="map-legend-copy">
                            <div class="map-legend-label">Thicker line</div>
                            <div class="map-legend-note">Higher selected metric value</div>
                        </div>
                    </div>
                    <div class="map-legend-item">
                        <span class="map-gradient-key"></span>
                        <div class="map-legend-copy">
                            <div class="map-legend-label">Darker route colour</div>
                            <div class="map-legend-note">Higher selected metric value</div>
                        </div>
                    </div>
                    <div class="map-legend-item">
                        <span class="map-endpoint-key"></span>
                        <div class="map-legend-copy">
                            <div class="map-legend-label">Green port marker</div>
                            <div class="map-legend-note">Endpoint of a visible route arc</div>
                        </div>
                    </div>
                    <div class="map-legend-item">
                        <span class="map-local-key"></span>
                        <div class="map-legend-copy">
                            <div class="map-legend-label">Copper port marker</div>
                            <div class="map-legend-note">Local short or same-port activity</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pydeck_route_map(route_summary: pd.DataFrame, title: str) -> None:
    """Render an arc-based route map on a real map background."""
    require_pydeck()

    if route_summary.empty:
        st.warning("No route data available for the selected period and filters.")
        return

    route_summary = route_summary.copy()
    route_summary["total_CO2_label"] = route_summary["total_CO2"].apply(
        lambda value: format_million_tonnes(kg_to_million_tonnes(value))
    )
    route_summary["weighted_CO2_per_km_label"] = route_summary["weighted_CO2_per_km"].apply(
        lambda value: "N/A" if pd.isna(value) else f"{value:,.2f}"
    )
    route_summary["weighted_CO2_per_dwt_km_label"] = route_summary["weighted_CO2_per_dwt_km"].apply(
        lambda value: "N/A" if pd.isna(value) else f"{value:,.6f}"
    )
    route_summary["voyage_count_label"] = route_summary["voyage_count"].apply(
        lambda value: f"{value:,.0f}"
    )
    route_summary["route_distance_km"] = approximate_route_distance_km(route_summary)
    arc_routes = route_summary[
        route_summary["route_distance_km"].fillna(0) >= 8
    ].copy()
    local_routes = route_summary[
        route_summary["route_distance_km"].fillna(0) < 8
    ].copy()
    arc_routes["hover_title"] = arc_routes["port_route_display"]
    arc_routes["hover_type"] = "Port-to-port route"
    arc_routes["hover_metric_title"] = arc_routes["map_metric_name"]
    arc_routes["hover_metric_value"] = (
        arc_routes["map_metric_label"] + " " + arc_routes["map_metric_unit"]
    )
    arc_routes["hover_total_co2_label"] = arc_routes["total_CO2_label"] + " million tonnes"
    arc_routes["hover_voyage_count_label"] = arc_routes["voyage_count_label"]
    arc_routes["hover_note_1"] = "Drawn as route arc"
    arc_routes["hover_note_2"] = "Hover shows aggregated route values"

    shadow_layer = pdk.Layer(
        "ArcLayer",
        data=arc_routes,
        get_source_position="[origin_lon, origin_lat]",
        get_target_position="[destination_lon, destination_lat]",
        get_source_color=[255, 253, 248, 235],
        get_target_color=[255, 253, 248, 235],
        get_width="shadow_width",
        pickable=False,
        parameters={"depthTest": False},
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_routes,
        get_source_position="[origin_lon, origin_lat]",
        get_target_position="[destination_lon, destination_lat]",
        get_source_color="route_color",
        get_target_color="route_color",
        get_width="line_width",
        pickable=True,
        auto_highlight=True,
        parameters={"depthTest": False},
    )

    def build_port_points(routes: pd.DataFrame, point_type: str) -> pd.DataFrame:
        """Build fixed-size port marker data for route endpoints or local activity."""
        if routes.empty:
            return pd.DataFrame(
                columns=[
                    "port",
                    "lat",
                    "lon",
                    "total_CO2",
                    "voyage_count",
                    "hover_title",
                    "hover_type",
                    "hover_metric_title",
                    "hover_metric_value",
                    "hover_total_co2_label",
                    "hover_voyage_count_label",
                    "hover_note_1",
                    "hover_note_2",
                ]
            )

        origin_points = routes.rename(
            columns={
                "origin_port_display": "port",
                "origin_lat": "lat",
                "origin_lon": "lon",
            }
        )
        destination_points = routes.rename(
            columns={
                "destination_port_display": "port",
                "destination_lat": "lat",
                "destination_lon": "lon",
            }
        )
        points = pd.concat(
            [
                origin_points[["port", "lat", "lon", "total_CO2", "voyage_count"]],
                destination_points[["port", "lat", "lon", "total_CO2", "voyage_count"]],
            ],
            ignore_index=True,
        )
        points = (
            points
            .groupby(["port", "lat", "lon"], dropna=False)
            .agg(total_CO2=("total_CO2", "sum"), voyage_count=("voyage_count", "sum"))
            .reset_index()
        )
        points["hover_title"] = points["port"]
        points["hover_type"] = point_type
        points["hover_metric_title"] = "Port-level activity"
        points["hover_metric_value"] = points["voyage_count"].apply(lambda value: f"{value:,.0f} voyages")
        points["hover_total_co2_label"] = points["total_CO2"].apply(
            lambda value: f"{format_million_tonnes(kg_to_million_tonnes(value))} million tonnes"
        )
        points["hover_voyage_count_label"] = points["voyage_count"].apply(lambda value: f"{value:,.0f}")
        if point_type == "Route endpoint":
            points["hover_note_1"] = "Endpoint of visible route arcs"
            points["hover_note_2"] = "Marker size is fixed for readability"
        else:
            points["hover_note_1"] = "Short or same-port movement"
            points["hover_note_2"] = "Represented at port level, not drawn as arc"
        return points

    endpoint_points = build_port_points(arc_routes, "Route endpoint")
    local_points = build_port_points(local_routes, "Local port activity")

    endpoint_outline_layer = pdk.Layer(
        "ScatterplotLayer",
        data=endpoint_points,
        get_position="[lon, lat]",
        get_radius=8,
        get_fill_color=[255, 255, 255, 210],
        stroked=False,
        filled=True,
        pickable=False,
        **{
            "radiusUnits": "pixels",
            "radiusMinPixels": 8,
            "radiusMaxPixels": 8,
        },
    )

    endpoint_core_layer = pdk.Layer(
        "ScatterplotLayer",
        data=endpoint_points,
        get_position="[lon, lat]",
        get_radius=6,
        get_fill_color=[31, 77, 58, 220],
        get_line_color=[255, 255, 255, 230],
        line_width_min_pixels=1.5,
        stroked=True,
        filled=True,
        pickable=True,
        **{
            "radiusUnits": "pixels",
            "radiusMinPixels": 6,
            "radiusMaxPixels": 6,
        },
    )

    local_outline_layer = pdk.Layer(
        "ScatterplotLayer",
        data=local_points,
        get_position="[lon, lat]",
        get_radius=8,
        get_fill_color=[255, 255, 255, 220],
        stroked=False,
        filled=True,
        pickable=False,
        **{
            "radiusUnits": "pixels",
            "radiusMinPixels": 8,
            "radiusMaxPixels": 8,
        },
    )

    local_core_layer = pdk.Layer(
        "ScatterplotLayer",
        data=local_points,
        get_position="[lon, lat]",
        get_radius=6,
        get_fill_color=[169, 91, 55, 225],
        get_line_color=[255, 255, 255, 235],
        line_width_min_pixels=1.5,
        stroked=True,
        filled=True,
        pickable=True,
        **{
            "radiusUnits": "pixels",
            "radiusMinPixels": 6,
            "radiusMaxPixels": 6,
        },
    )

    deck = pdk.Deck(
        layers=[
            shadow_layer,
            arc_layer,
            local_outline_layer,
            local_core_layer,
            endpoint_outline_layer,
            endpoint_core_layer,
        ],
        initial_view_state=get_initial_view(route_summary),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": (
                "<div style='font-family:Avenir Next,Arial,sans-serif;min-width:230px;'>"
                "<div style='font-weight:700;font-size:13px;margin-bottom:6px;color:#1F2A24;'>"
                "{hover_title}</div>"
                "<div style='height:1px;background:#D7E0D5;margin:5px 0 7px 0;'></div>"
                "<div style='font-size:12px;color:#657369;'>{hover_type}</div>"
                "<div style='font-size:13px;font-weight:700;color:#1F4D3A;margin-bottom:5px;'>"
                "{hover_metric_title}: {hover_metric_value}</div>"
                "<div style='font-size:12px;color:#1F2A24;'>Total CO₂: <b>{hover_total_co2_label}</b></div>"
                "<div style='font-size:12px;color:#1F2A24;'>Voyages: <b>{hover_voyage_count_label}</b></div>"
                "<div style='font-size:12px;color:#1F2A24;margin-top:5px;'>{hover_note_1}</div>"
                "<div style='font-size:12px;color:#657369;'>{hover_note_2}</div>"
                "</div>"
            ),
            "style": {
                "backgroundColor": "rgba(255, 255, 255, 0.96)",
                "border": "1px solid #D7E0D5",
                "borderRadius": "8px",
                "boxShadow": "0 10px 24px rgba(31, 42, 36, 0.16)",
                "color": "#1F2A24",
                "padding": "10px",
            },
        },
    )

    metric_name = (
        route_summary["map_metric_name"].iloc[0]
        if "map_metric_name" in route_summary.columns and not route_summary.empty
        else "selected metric"
    )
    metric_unit = (
        route_summary["map_metric_unit"].iloc[0]
        if "map_metric_unit" in route_summary.columns and not route_summary.empty
        else ""
    )
    metric_context_label = f"{metric_name} ({metric_unit})" if metric_unit else metric_name
    route_count = len(route_summary)
    arc_count = len(arc_routes)
    st.markdown(f"#### {title}")
    render_map_context_panel(metric_context_label, route_count, arc_count)
    st.caption(
        "This map combines route flows and port activity. Very short or same-port movements remain in the selected data "
        "but are represented at port level rather than drawn as arcs, to avoid misleading loop shapes."
    )
    st.pydeck_chart(deck, use_container_width=True, height=650)


# ============================================================
# 6. KPI cards
# ============================================================
# This section calculates headline metrics such as total CO2,
# voyage count, vessel count, total distance, and average CO2/km.
# ============================================================

def render_kpi_cards(kpis: list[tuple[str, str]], compact: bool = False) -> None:
    """Render KPI cards with the dashboard typography and spacing."""
    grid_class = "kpi-grid kpi-grid-compact" if compact else "kpi-grid"
    cards = "".join(
        (
            '<div class="kpi-card">'
            f'<div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div>'
            "</div>"
        )
        for label, value in kpis
    )

    st.markdown(
        f'<div class="{grid_class}">{cards}</div>',
        unsafe_allow_html=True,
    )


def show_kpis(df: pd.DataFrame) -> None:
    """Show headline KPI cards."""
    total_co2 = df["CO2"].sum()
    voyage_count = len(df)
    vessel_count = df["imo"].nunique()
    total_distance = df["delta_dist_km"].sum()
    # Network-level intensity should be distance weighted. Summing emissions and
    # distance gives long and short voyages influence in proportion to the
    # distance they represent, instead of treating every voyage equally.
    intensity_valid = (
        df["CO2"].notna()
        & df["delta_dist_km"].notna()
        & (df["delta_dist_km"] > 0)
    )
    valid_total_co2 = df.loc[intensity_valid, "CO2"].sum()
    valid_total_distance = df.loc[intensity_valid, "delta_dist_km"].sum()
    weighted_co2_per_km = (
        valid_total_co2 / valid_total_distance
        if valid_total_distance > 0
        else np.nan
    )

    dwt_intensity_valid = (
        df["CO2"].notna()
        & df["delta_dist_km"].notna()
        & (df["delta_dist_km"] > 0)
        & df["dwt"].notna()
        & (df["dwt"] > 0)
    )
    valid_total_dwt_km = (
        df.loc[dwt_intensity_valid, "delta_dist_km"]
        * df.loc[dwt_intensity_valid, "dwt"]
    ).sum()
    weighted_co2_per_dwt_km = (
        df.loc[dwt_intensity_valid, "CO2"].sum() / valid_total_dwt_km
        if valid_total_dwt_km > 0
        else np.nan
    )

    kpis = [
        ("Total CO₂ (million tonnes)", format_million_tonnes(kg_to_million_tonnes(total_co2))),
        ("Voyages (records)", format_number(voyage_count)),
        ("Vessels (unique IMO)", format_number(vessel_count)),
        ("Distance (km)", format_number(total_distance)),
        (
            "Distance-weighted CO₂ intensity (kg/km)",
            f"{weighted_co2_per_km:,.2f}" if pd.notna(weighted_co2_per_km) else "N/A",
        ),
        (
            "Capacity-adjusted CO₂ intensity (kg/DWT-km)",
            f"{weighted_co2_per_dwt_km:.6f}"
            if pd.notna(weighted_co2_per_dwt_km)
            else "N/A",
        ),
    ]
    render_kpi_cards(kpis)


# 7. Overview page

SCENARIO_EXCLUDED_SHIP_TYPES = {
    "Unknown",
    "Other activities",
    "Other ship types",
}


def reset_decarbonisation_scenario() -> None:
    """Return all illustrative scenario controls to their baseline values."""
    for index in range(1, 7):
        st.session_state[f"scenario_ship_reduction_{index}"] = 0


def show_decarbonisation_scenario(df: pd.DataFrame) -> None:
    """Render a simple Unity-aligned, filter-responsive what-if scenario."""
    baseline_co2 = df["CO2"].fillna(0).sum()
    if baseline_co2 <= 0:
        st.info("No positive CO₂ values are available for a decarbonisation scenario.")
        return

    ship_emissions = (
        df.assign(ship_type_clean=df["ship_type_clean"].fillna("Unknown").astype(str))
        .groupby("ship_type_clean", dropna=False)["CO2"]
        .sum()
        .sort_values(ascending=False)
    )
    eligible_ship_emissions = ship_emissions[
        ~ship_emissions.index.isin(SCENARIO_EXCLUDED_SHIP_TYPES)
        & (ship_emissions > 0)
    ].head(6)
    if eligible_ship_emissions.empty:
        st.info("No identifiable ship type is available for this scenario.")
        return

    top_ship_signature = tuple(eligible_ship_emissions.index.astype(str))
    if st.session_state.get("scenario_top_ship_signature") != top_ship_signature:
        reset_decarbonisation_scenario()
        st.session_state["scenario_top_ship_signature"] = top_ship_signature

    st.markdown("### Illustrative decarbonisation scenario")
    st.caption(
        "Adjust simple what-if assumptions for the current filtered data. "
        "The six ship types are selected automatically from the highest-emitting "
        "identifiable categories. This is an illustrative comparison, not a forecast."
    )

    st.button(
        "Reset scenario",
        on_click=reset_decarbonisation_scenario,
    )

    reduction_rates: list[float] = []
    top_ship_rows = list(eligible_ship_emissions.items())
    ship_rows = [top_ship_rows[:3], top_ship_rows[3:]]
    ship_index = 0
    for row in ship_rows:
        if not row:
            continue
        control_columns = st.columns(len(row))
        for column, (ship_type, ship_co2) in zip(control_columns, row):
            with column:
                ship_share = ship_co2 / baseline_co2 * 100
                reduction_percent = st.slider(
                    f"{ship_type} CO₂ reduction",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    key=f"scenario_ship_reduction_{ship_index + 1}",
                    help=(
                        f"{ship_type} currently contributes {ship_share:.1f}% of CO₂ "
                        "within the selected data."
                    ),
                )
                st.caption(
                    f"Current contribution: "
                    f"{format_million_tonnes(kg_to_million_tonnes(ship_co2))} million tonnes "
                    f"({ship_share:.1f}%)"
                )
                reduction_rates.append(reduction_percent / 100)
                ship_index += 1

    ship_avoided_values = [
        ship_co2 * reduction_rate
        for (_, ship_co2), reduction_rate in zip(top_ship_rows, reduction_rates)
    ]
    ship_avoided_co2 = sum(ship_avoided_values)
    avoided_co2 = min(baseline_co2, ship_avoided_co2)
    scenario_co2 = baseline_co2 - avoided_co2
    overall_reduction_percent = avoided_co2 / baseline_co2 * 100

    render_kpi_cards(
        [
            ("Baseline CO₂ (million tonnes)", format_million_tonnes(kg_to_million_tonnes(baseline_co2))),
            ("Avoided CO₂ (million tonnes)", format_million_tonnes(kg_to_million_tonnes(avoided_co2))),
            ("Scenario CO₂ (million tonnes)", format_million_tonnes(kg_to_million_tonnes(scenario_co2))),
            ("Potential reduction", f"{overall_reduction_percent:.1f}%"),
        ],
        compact=True,
    )


def show_overview(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> None:
    """Render the overview page."""
    st.subheader("Overview")
    show_kpis(df)

    reference_df = reference_df if reference_df is not None else df

    monthly = (
        df.groupby("departure_month", dropna=False)
        .agg(
            total_CO2=("CO2", "sum"),
            voyage_count=("CO2", "size"),
        )
        .reset_index()
        .sort_values("departure_month")
    )
    monthly["total_CO2"] = monthly["total_CO2"] / KG_PER_MILLION_TONNES
    monthly["departure_month_plot"] = pd.to_datetime(
        monthly["departure_month"], format="%Y-%m", errors="coerce"
    )

    unique_port_routes = df["port_route_display"].dropna().astype(str).unique()
    unique_country_pairs = df["country_pair"].dropna().astype(str).unique()
    selected_route_col = None
    selected_route_label = None

    if len(unique_port_routes) == 1:
        selected_route_col = "port_route_display"
        selected_route_label = unique_port_routes[0]
    elif len(unique_country_pairs) == 1:
        selected_route_col = "country_pair"
        selected_route_label = unique_country_pairs[0]

    if selected_route_col is not None:
        st.markdown(f"#### What does this selected route look like? {selected_route_label}")

        route_totals = (
            reference_df.groupby(selected_route_col, dropna=False)["CO2"]
            .sum()
            .sort_values(ascending=False)
        )
        route_rank = (
            int(route_totals.index.astype(str).tolist().index(str(selected_route_label)) + 1)
            if str(selected_route_label) in route_totals.index.astype(str).tolist()
            else None
        )
        total_reference_co2 = reference_df["CO2"].sum()
        selected_co2 = df["CO2"].sum()
        selected_share = (
            selected_co2 / total_reference_co2 * 100
            if total_reference_co2 > 0
            else np.nan
        )
        dominant_scope = (
            df["voyage_scope"].dropna().astype(str).mode().iloc[0]
            if not df["voyage_scope"].dropna().empty
            else "Unknown"
        )

        render_kpi_cards(
            [
                (
                    f"Route rank by CO₂ among {selected_route_col.replace('_', ' ')}",
                    f"#{route_rank}" if route_rank is not None else "N/A",
                ),
                (
                    "Share of filtered dataset CO₂",
                    f"{selected_share:.2f}%" if pd.notna(selected_share) else "N/A",
                ),
                ("Route scope", dominant_scope),
            ],
            compact=True,
        )

        ship_breakdown = (
            df.groupby("ship_type_clean", dropna=False)
            .agg(
                total_CO2=("CO2", "sum"),
                voyage_count=("CO2", "size"),
            )
            .reset_index()
            .sort_values("total_CO2", ascending=False)
            .head(12)
        )
        ship_breakdown["total_CO2"] = ship_breakdown["total_CO2"] / KG_PER_MILLION_TONNES

        components = pd.DataFrame(
            {
                "Component": ["Main engine", "Auxiliary engine", "Boiler"],
                "CO2": [
                    df["CO2_ME_kg"].sum(),
                    df["CO2_AE_kg"].sum(),
                    df["CO2_boiler_kg"].sum(),
                ],
            }
        )
        components["Share (%)"] = components["CO2"] / components["CO2"].sum() * 100
        components["CO2_million_tonnes"] = components.pop("CO2") / KG_PER_MILLION_TONNES

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                monthly,
                x="departure_month_plot",
                y="total_CO2",
                markers=True,
                title="How do emissions change over time on this route?",
                labels={
                    "departure_month_plot": "Departure month",
                    "total_CO2": "Total CO₂ (million tonnes)",
                },
            )
            fig.update_xaxes(dtick="M6", tickformat="%b<br>%Y")
            render_plotly_chart(fig)

        with col2:
            ship_plot = ship_breakdown.sort_values("total_CO2", ascending=False)
            fig = px.bar(
                ship_plot,
                x="total_CO2",
                y="ship_type_clean",
                orientation="h",
                title="Which ship types drive emissions on this route?",
                labels={
                    "total_CO2": "Total CO₂ (million tonnes)",
                    "ship_type_clean": "Ship Type",
                },
                category_orders={
                    "ship_type_clean": ship_plot["ship_type_clean"].tolist()
                },
            )
            render_plotly_chart(fig)

        col1, col2 = st.columns(2)

        with col1:
            valid_voyages = df[df["CO2"] > 0]
            if not valid_voyages.empty:
                fig = px.histogram(
                    valid_voyages,
                    x="CO2",
                    nbins=40,
                    title="How variable are voyage-level emissions?",
                    labels={"CO2": "CO₂ per voyage (kg)"},
                )
                fig.update_yaxes(title_text="Voyages (records)")
                render_plotly_chart(fig)
            else:
                st.info("No positive CO₂ values are available for the selected route.")

        with col2:
            fig = px.pie(
                components,
                values="CO2_million_tonnes",
                names="Component",
                title="Which engine source contributes most?",
                hole=0.35,
                labels={"CO2_million_tonnes": "CO₂ (million tonnes)"},
            )
            fig = style_pie_chart(fig)
            fig.update_traces(
                hovertemplate=(
                    "<b>%{label}</b><br>CO₂: %{value:.2f} million tonnes"
                    "<br>Share: %{percent:.1%}<extra></extra>"
                )
            )
            render_plotly_chart(fig)

        with st.expander("Route-level data table"):
            route_table = (
                df.groupby(
                    [
                        "departure_month",
                        "ship_type_clean",
                    ],
                    dropna=False,
                )
                .agg(
                    total_CO2=("CO2", "sum"),
                    total_distance_km=("delta_dist_km", "sum"),
                    voyage_count=("CO2", "size"),
                )
                .reset_index()
                .sort_values(["departure_month", "total_CO2"], ascending=[True, False])
            )
            route_table["weighted_CO2_per_km"] = (
                route_table["total_CO2"]
                / route_table["total_distance_km"].replace(0, np.nan)
            )
            display_dataframe(route_table, use_container_width=True, hide_index=True)

        show_decarbonisation_scenario(df)
        return

    scope = (
        df.groupby("voyage_scope", dropna=False)
        .agg(
            total_CO2=("CO2", "sum"),
            voyage_count=("CO2", "size"),
        )
        .reset_index()
    )
    scope["total_CO2"] = scope["total_CO2"] / KG_PER_MILLION_TONNES

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            monthly,
            x="departure_month_plot",
            y="total_CO2",
            markers=True,
            title="How do emissions change over time?",
            labels={
                "departure_month_plot": "Departure month",
                "total_CO2": "Total CO₂ (million tonnes)",
            },
        )
        fig.update_xaxes(dtick="M6", tickformat="%b<br>%Y")
        render_plotly_chart(fig)

    with col2:
        fig = px.bar(
            scope,
            x="voyage_scope",
            y="total_CO2",
            color="voyage_scope",
            title="Are emissions mainly domestic or international?",
            labels={"voyage_scope": "Voyage Scope", "total_CO2": "Total CO₂ (million tonnes)"},
        )
        render_plotly_chart(fig)

    components = pd.DataFrame(
        {
            "Component": ["Main engine", "Auxiliary engine", "Boiler"],
            "CO2": [
                df["CO2_ME_kg"].sum(),
                df["CO2_AE_kg"].sum(),
                df["CO2_boiler_kg"].sum(),
            ],
        }
    )

    components["Share (%)"] = components["CO2"] / components["CO2"].sum() * 100
    components["CO2_million_tonnes"] = components.pop("CO2") / KG_PER_MILLION_TONNES

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            components,
            values="CO2_million_tonnes",
            names="Component",
            title="Which engine source contributes most?",
            hole=0.35,
            labels={"CO2_million_tonnes": "CO₂ (million tonnes)"},
        )
        fig = style_pie_chart(fig)
        fig.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>CO₂: %{value:.2f} million tonnes"
                "<br>Share: %{percent:.1%}<extra></extra>"
            )
        )
        render_plotly_chart(fig)

    with col2:
        display_dataframe(components, use_container_width=True, hide_index=True)

    show_decarbonisation_scenario(df)


# 8. Route explorer page

def route_evidence_threshold(route_col: str) -> int:
    """Return a minimum activity threshold for stable route comparisons."""
    return {
        "country_pair": 100,
        "route_directional": 50,
        "port_route_display": 20,
    }.get(route_col, 20)


def build_route_operational_summary(df: pd.DataFrame, route_col: str) -> pd.DataFrame:
    """Build route totals and correctly weighted intensity metrics."""
    base = (
        df.groupby(route_col, dropna=False)
        .agg(
            voyage_count=("CO2", "size"),
            total_CO2=("CO2", "sum"),
        )
        .reset_index()
    )

    distance_valid = df[
        df["CO2"].notna()
        & df["delta_dist_km"].notna()
        & (df["delta_dist_km"] > 0)
    ]
    distance_summary = (
        distance_valid.groupby(route_col, dropna=False)
        .agg(
            distance_CO2=("CO2", "sum"),
            total_distance_km=("delta_dist_km", "sum"),
        )
        .reset_index()
    )
    distance_summary["weighted_CO2_per_km"] = (
        distance_summary["distance_CO2"] / distance_summary["total_distance_km"]
    )

    dwt_valid = distance_valid[
        distance_valid["dwt"].notna() & (distance_valid["dwt"] > 0)
    ].copy()
    dwt_valid["dwt_km"] = dwt_valid["delta_dist_km"] * dwt_valid["dwt"]
    dwt_summary = (
        dwt_valid.groupby(route_col, dropna=False)
        .agg(
            dwt_CO2=("CO2", "sum"),
            total_dwt_km=("dwt_km", "sum"),
            mean_dwt=("dwt", "mean"),
        )
        .reset_index()
    )
    dwt_summary["weighted_CO2_per_dwt_km"] = (
        dwt_summary["dwt_CO2"] / dwt_summary["total_dwt_km"]
    )

    summary = base.merge(distance_summary, on=route_col, how="left").merge(
        dwt_summary, on=route_col, how="left"
    )
    summary["total_CO2_million_tonnes"] = summary["total_CO2"] / KG_PER_MILLION_TONNES
    return summary


def show_route_explorer(
    df: pd.DataFrame,
    metric_name: str,
    top_n: int,
    route_mode: str,
) -> None:
    """Render country-pair and route analysis."""
    st.subheader("Route Explorer")

    if route_mode == "Country pair":
        route_col = "country_pair"
        route_label = "country pairs"
    elif route_mode == "Directional country route":
        route_col = "route_directional"
        route_label = "directional flows"
    else:
        route_col = "port_route_display"
        route_label = "port-to-port movements"

    metric_col = metric_column(metric_name)
    metric_label = format_metric_label(metric_name)
    minimum_route_voyages = (
        route_evidence_threshold(route_col)
        if metric_name in {"CO2 per km", "CO2 per DWT-km"}
        else 0
    )
    route_summary = aggregate_metric(
        df,
        route_col,
        metric_name,
        top_n,
        min_voyages=minimum_route_voyages,
    )
    unique_routes = df[route_col].dropna().astype(str).nunique()

    if unique_routes <= 1 and not route_summary.empty:
        st.caption(
            "Use this view to drill into the selected route and inspect its internal movements, vessels, and voyage-level variation."
        )
    else:
        st.caption(
            f"Use this view to compare {route_label} under the selected ranking metric."
        )

    if unique_routes <= 1 and not route_summary.empty:
        selected_route = str(route_summary[route_col].iloc[0])
        metric_value = route_summary[metric_col].iloc[0]
        total_co2 = df["CO2"].sum()
        voyage_count = len(df)
        vessel_count = df["imo"].nunique()
        port_route_count = df["port_route_display"].dropna().astype(str).nunique()
        direction_count = df["route_directional"].dropna().astype(str).nunique()

        st.markdown(f"#### Route drill-down: {selected_route}")
        st.caption(
            "A single route is selected, so this view breaks down its internal movements and vessel contributors "
            "instead of ranking the route against itself."
        )

        render_kpi_cards(
            [
                (f"Selected route {metric_label}", format_metric_value(metric_name, metric_value)),
                ("Total CO₂ (million tonnes)", format_million_tonnes(kg_to_million_tonnes(total_co2))),
                ("Voyages (records)", format_number(voyage_count)),
                ("Vessels (unique IMO)", format_number(vessel_count)),
                ("Port-to-port movements", format_number(port_route_count)),
            ],
            compact=True,
        )

        route_df = df.copy()
        route_df["vessel_label"] = np.where(
            route_df["vessel_name"].notna(),
            route_df["vessel_name"].astype(str) + " (" + route_df["imo"].astype(str) + ")",
            "IMO " + route_df["imo"].astype(str),
        )

        if port_route_count > 1:
            left_summary = aggregate_metric(route_df, "port_route_display", metric_name, top_n)
            left_group_col = "port_route_display"
            left_title = f"Which port-to-port movements make up this selection by {metric_label}?"
            left_axis = "Port-to-port movement"
        elif direction_count > 1:
            left_summary = aggregate_metric(route_df, "route_directional", metric_name, top_n)
            left_group_col = "route_directional"
            left_title = f"How is this selection split by direction using {metric_label}?"
            left_axis = "Directional flow"
        else:
            left_summary = aggregate_metric(route_df, "ship_type_clean", metric_name, top_n)
            left_group_col = "ship_type_clean"
            left_title = f"Which ship types contribute within this route by {metric_label}?"
            left_axis = "Ship Type"

        vessel_summary = aggregate_metric(route_df, "vessel_label", metric_name, min(top_n, 20))

        col1, col2 = st.columns([1.15, 1])

        with col1:
            fig = px.bar(
                left_summary.sort_values(metric_col, ascending=True),
                x=metric_col,
                y=left_group_col,
                orientation="h",
                title=left_title,
                labels={metric_col: metric_label, left_group_col: left_axis},
            )
            render_plotly_chart(fig)

        with col2:
            fig = px.bar(
                vessel_summary.sort_values(metric_col, ascending=True),
                x=metric_col,
                y="vessel_label",
                orientation="h",
                title=f"Which vessels contribute most by {metric_label}?",
                labels={metric_col: metric_label, "vessel_label": "Vessel"},
            )
            render_plotly_chart(fig)

        valid_voyages = route_df[
            route_df["CO2"].notna()
            & route_df["delta_dist_km"].notna()
            & (route_df["delta_dist_km"] > 0)
        ]
        if not valid_voyages.empty:
            scatter_df = valid_voyages.sample(
                n=min(len(valid_voyages), 5000),
                random_state=42,
            )
            fig = px.scatter(
                scatter_df,
                x="delta_dist_km",
                y="CO2",
                color="ship_type_clean",
                hover_data=[
                    "vessel_name",
                    "origin_port_display",
                    "destination_port_display",
                    "departure_time",
                    "CO2_per_km",
                ],
                title="Do individual voyages on this route behave differently?",
                labels={
                    "delta_dist_km": "Distance (km)",
                    "CO2": "CO₂ per voyage (kg)",
                    "ship_type_clean": "Ship Type",
                    "CO2_per_km": "CO₂ intensity (kg/km)",
                },
            )
            render_plotly_chart(fig, height=460)

        route_detail_source = route_df.copy()
        route_detail_dwt_valid = (
            route_detail_source["CO2"].notna()
            & route_detail_source["delta_dist_km"].notna()
            & (route_detail_source["delta_dist_km"] > 0)
            & route_detail_source["dwt"].notna()
            & (route_detail_source["dwt"] > 0)
        )
        route_detail_source["dwt_valid_CO2"] = route_detail_source["CO2"].where(
            route_detail_dwt_valid
        )
        route_detail_source["total_dwt_km"] = (
            route_detail_source["delta_dist_km"] * route_detail_source["dwt"]
        ).where(route_detail_dwt_valid)
        route_detail = (
            route_detail_source.groupby(["route_directional", "port_route_display", "ship_type_clean"], dropna=False)
            .agg(
                voyage_count=("CO2", "size"),
                total_CO2=("CO2", "sum"),
                vessels=("imo", "nunique"),
                distance_km=("delta_dist_km", "sum"),
                dwt_valid_CO2=("dwt_valid_CO2", "sum"),
                total_dwt_km=("total_dwt_km", "sum"),
            )
            .reset_index()
            .sort_values("total_CO2", ascending=False)
            .head(80)
        )
        route_detail["weighted_CO2_per_km"] = (
            route_detail["total_CO2"]
            / route_detail["distance_km"].replace(0, np.nan)
        )
        route_detail["weighted_CO2_per_dwt_km"] = (
            route_detail["dwt_valid_CO2"]
            / route_detail["total_dwt_km"].replace(0, np.nan)
        )
        with st.expander("Route drill-down table"):
            display_dataframe(route_detail, use_container_width=True, hide_index=True)
        return

    col1, col2 = st.columns([1.3, 1])

    with col1:
        route_plot = route_summary.sort_values(metric_col, ascending=True).copy()
        route_plot["value_label"] = route_plot[metric_col].apply(
            lambda value: (
                f"{format_metric_value(metric_name, value)} Mt"
                if metric_name == "Total CO2"
                else format_metric_value(metric_name, value)
            )
        )
        route_hover_format = {
            "CO2 per DWT-km": ".6f",
            "Voyage count": ",.0f",
        }.get(metric_name, ".2f")
        fig = px.bar(
            route_plot,
            x=metric_col,
            y=route_col,
            orientation="h",
            title=f"Which routes rank highest by {metric_label}?",
            labels={metric_col: metric_label, route_col: "Route"},
            text="value_label",
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_color=CLIMATE_COLORS["blue"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                + metric_label
                + f": %{{x:{route_hover_format}}}<extra></extra>"
            ),
        )
        route_axis_max = route_plot[metric_col].max()
        fig.update_xaxes(
            range=[0, route_axis_max * 1.24] if route_axis_max > 0 else None,
        )
        if metric_name == "Total CO2":
            fig.update_xaxes(tickformat=".2f")
        render_plotly_chart(fig)

    with col2:
        display_dataframe(route_summary, use_container_width=True, hide_index=True)

    if metric_name == "Total CO2":
        st.markdown("#### What drives emissions on the leading routes?")
        st.caption(
            "Compare the ship-type composition of the ten highest-emission routes, then select one route for a detailed contribution profile."
        )

        top_route_overview = route_summary.head(10).copy()
        top_route_overview["value_label"] = top_route_overview[metric_col].apply(
            lambda value: f"{format_million_tonnes(value)} Mt"
        )
        top_routes = top_route_overview[route_col].astype(str).tolist()

        top_route_mix_source = df[df[route_col].astype(str).isin(top_routes)].copy()
        top_route_mix_source["ship_type_clean"] = (
            top_route_mix_source["ship_type_clean"].fillna("Unknown").astype(str)
        )
        leading_mix_ship_types = (
            top_route_mix_source.groupby("ship_type_clean", dropna=False)["CO2"]
            .sum()
            .sort_values(ascending=False)
            .head(6)
            .index.astype(str)
            .tolist()
        )
        top_route_mix_source["ship_type_display"] = np.where(
            top_route_mix_source["ship_type_clean"].isin(leading_mix_ship_types),
            top_route_mix_source["ship_type_clean"],
            "Other ship types",
        )
        top_route_mix = (
            top_route_mix_source.groupby(
                [route_col, "ship_type_display"], dropna=False
            )["CO2"]
            .sum()
            .reset_index(name="total_CO2")
        )
        top_route_mix["total_CO2_million_tonnes"] = (
            top_route_mix["total_CO2"] / KG_PER_MILLION_TONNES
        )

        overview_col, detail_col = st.columns([1.05, 1])

        with overview_col:
            fig = px.bar(
                top_route_mix,
                x="total_CO2_million_tonnes",
                y=route_col,
                color="ship_type_display",
                orientation="h",
                title="Which ship types shape the top 10 routes?",
                labels={
                    "total_CO2_million_tonnes": "Total CO₂ (million tonnes)",
                    route_col: "Route",
                    "ship_type_display": "Ship type",
                },
                category_orders={route_col: top_routes},
                color_discrete_sequence=CLIMATE_SEQUENCE,
            )
            fig.update_traces(
                offsetgroup="ship_type_mix",
                hovertemplate=(
                    "<b>%{y}</b><br>%{fullData.name}"
                    "<br>CO₂: %{x:.2f} million tonnes<extra></extra>"
                ),
            )
            fig.update_xaxes(tickformat=".2f")
            fig.update_layout(barmode="stack", legend_title_text="Ship type")
            render_plotly_chart(fig, height=520, extra_top_margin=34)

        with detail_col:
            route_display_func = format_country_pair_option if route_col == "country_pair" else str
            selected_driver_route = st.selectbox(
                "Route to inspect",
                top_routes,
                format_func=route_display_func,
                key=f"route_driver_selection_{route_mode}",
                help="Choose one of the ten leading routes to inspect its ship-type composition.",
            )

            ship_breakdown = (
                df[df[route_col].astype(str).eq(selected_driver_route)]
                .groupby("ship_type_clean", dropna=False)["CO2"]
                .sum()
                .reset_index(name="total_CO2")
                .sort_values("total_CO2", ascending=False)
            )
            ship_breakdown["ship_type_clean"] = (
                ship_breakdown["ship_type_clean"].fillna("Unknown").astype(str)
            )
            leading_ship_types = ship_breakdown.head(6).copy()
            other_total = ship_breakdown.iloc[6:]["total_CO2"].sum()
            if other_total > 0:
                leading_ship_types = pd.concat(
                    [
                        leading_ship_types,
                        pd.DataFrame(
                            {"ship_type_clean": ["Other ship types"], "total_CO2": [other_total]}
                        ),
                    ],
                    ignore_index=True,
                )

            route_total = leading_ship_types["total_CO2"].sum()
            leading_ship_types["total_CO2_million_tonnes"] = (
                leading_ship_types["total_CO2"] / KG_PER_MILLION_TONNES
            )
            leading_ship_types["share_pct"] = np.where(
                route_total > 0,
                leading_ship_types["total_CO2"] / route_total * 100,
                np.nan,
            )
            leading_ship_types["value_label"] = leading_ship_types.apply(
                lambda row: (
                    f"{format_million_tonnes(row['total_CO2_million_tonnes'])} Mt · "
                    f"{row['share_pct']:.1f}%"
                ),
                axis=1,
            )
            ship_plot = leading_ship_types.sort_values(
                "total_CO2_million_tonnes", ascending=True
            )
            bar_colors = [
                CLIMATE_COLORS["muted"]
                if ship_type in {"Unknown", "Other ship types"}
                else CLIMATE_COLORS["teal"]
                for ship_type in ship_plot["ship_type_clean"]
            ]

            fig = px.bar(
                ship_plot,
                x="total_CO2_million_tonnes",
                y="ship_type_clean",
                orientation="h",
                title=f"Ship-type contribution: {selected_driver_route}",
                labels={
                    "total_CO2_million_tonnes": "Total CO₂ (million tonnes)",
                    "ship_type_clean": "Ship Type",
                },
                text="value_label",
                custom_data=["share_pct"],
            )
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{y}</b><br>Total CO₂: %{x:.2f} million tonnes"
                    "<br>Share of route: %{customdata[0]:.1f}%<extra></extra>"
                ),
            )
            add_ship_type_colour_key(fig, CLIMATE_COLORS["teal"])
            ship_axis_max = ship_plot["total_CO2_million_tonnes"].max()
            fig.update_xaxes(
                tickformat=".2f",
                range=[0, ship_axis_max * 1.80] if ship_axis_max > 0 else None,
            )
            render_plotly_chart(fig, height=500, extra_top_margin=34)

    elif metric_name == "CO2 per km":
        st.markdown("#### Where do scale and emissions intensity coincide?")
        minimum_voyages = route_evidence_threshold(route_col)
        st.caption(
            f"Routes require at least {minimum_voyages} voyages. Weighted intensity uses total CO₂ divided by total distance, "
            "so high-volume routes do not receive the same weight as short, isolated voyages."
        )
        route_analytics = build_route_operational_summary(df, route_col)
        qualified = route_analytics[
            (route_analytics["voyage_count"] >= minimum_voyages)
            & route_analytics["weighted_CO2_per_km"].notna()
            & (route_analytics["weighted_CO2_per_km"] > 0)
            & (route_analytics["total_CO2_million_tonnes"] > 0)
        ].copy()

        if qualified.empty:
            st.info("Not enough route activity is available for a stable weighted-intensity comparison.")
        else:
            emissions_midpoint = qualified["total_CO2_million_tonnes"].median()
            intensity_midpoint = qualified["weighted_CO2_per_km"].median()
            qualified["Priority profile"] = np.select(
                [
                    (qualified["total_CO2_million_tonnes"] >= emissions_midpoint)
                    & (qualified["weighted_CO2_per_km"] >= intensity_midpoint),
                    qualified["total_CO2_million_tonnes"] >= emissions_midpoint,
                    qualified["weighted_CO2_per_km"] >= intensity_midpoint,
                ],
                [
                    "High emissions · High intensity",
                    "High emissions · Lower intensity",
                    "Lower emissions · High intensity",
                ],
                default="Lower emissions · Lower intensity",
            )

            quadrant_col, trend_col = st.columns([1.2, 1])
            with quadrant_col:
                fig = px.scatter(
                    qualified,
                    x="total_CO2_million_tonnes",
                    y="weighted_CO2_per_km",
                    color="Priority profile",
                    size="voyage_count",
                    size_max=20,
                    hover_name=route_col,
                    hover_data={
                        "total_CO2_million_tonnes": ":.2f",
                        "weighted_CO2_per_km": ":.2f",
                        "voyage_count": ":,.0f",
                        "Priority profile": False,
                    },
                    log_x=True,
                    log_y=True,
                    title="Route priority matrix",
                    labels={
                        "total_CO2_million_tonnes": "Total CO₂ (million tonnes, log scale)",
                        "weighted_CO2_per_km": "Distance-weighted CO₂ intensity (kg/km, log scale)",
                        "voyage_count": "Voyages",
                    },
                    color_discrete_map={
                        "High emissions · High intensity": CLIMATE_COLORS["coral"],
                        "High emissions · Lower intensity": CLIMATE_COLORS["amber"],
                        "Lower emissions · High intensity": CLIMATE_COLORS["purple"],
                        "Lower emissions · Lower intensity": CLIMATE_COLORS["teal"],
                    },
                )
                fig.update_traces(
                    marker={
                        "opacity": 0.8,
                        "sizemin": 5,
                        "line": {"color": "#FFFFFF", "width": 1.2},
                    }
                )
                fig.update_xaxes(dtick=1)
                fig.update_yaxes(dtick=1)
                fig.add_vline(
                    x=emissions_midpoint,
                    line_dash="dot",
                    line_color=CLIMATE_COLORS["muted"],
                )
                fig.add_hline(
                    y=intensity_midpoint,
                    line_dash="dot",
                    line_color=CLIMATE_COLORS["muted"],
                )
                x_min = qualified["total_CO2_million_tonnes"].min()
                x_max = qualified["total_CO2_million_tonnes"].max()
                y_min = qualified["weighted_CO2_per_km"].min()
                y_max = qualified["weighted_CO2_per_km"].max()
                x_span = np.log10(x_max) - np.log10(x_min)
                y_span = np.log10(y_max) - np.log10(y_min)
                x_boundary = (
                    (np.log10(emissions_midpoint) - np.log10(x_min)) / x_span
                    if x_span > 0
                    else 0.5
                )
                y_boundary = (
                    (np.log10(intensity_midpoint) - np.log10(y_min)) / y_span
                    if y_span > 0
                    else 0.5
                )
                quadrant_annotations = [
                    (
                        max(x_boundary / 2, 0.13),
                        min((y_boundary + 1) / 2, 0.87),
                        "Lower emissions<br>High intensity",
                        CLIMATE_COLORS["purple"],
                    ),
                    (
                        min((x_boundary + 1) / 2, 0.87),
                        min((y_boundary + 1) / 2, 0.87),
                        "High emissions<br>High intensity",
                        CLIMATE_COLORS["coral"],
                    ),
                    (
                        max(x_boundary / 2, 0.13),
                        max(y_boundary / 2, 0.13),
                        "Lower emissions<br>Lower intensity",
                        CLIMATE_COLORS["teal"],
                    ),
                    (
                        min((x_boundary + 1) / 2, 0.87),
                        max(y_boundary / 2, 0.13),
                        "High emissions<br>Lower intensity",
                        CLIMATE_COLORS["amber"],
                    ),
                ]
                for x_position, y_position, label, border_color in quadrant_annotations:
                    fig.add_annotation(
                        x=x_position,
                        y=y_position,
                        xref="paper",
                        yref="paper",
                        text=label,
                        showarrow=False,
                        align="center",
                        font={"size": 9, "color": CLIMATE_COLORS["text"]},
                        bgcolor="rgba(255,255,255,0.84)",
                        bordercolor=border_color,
                        borderwidth=1,
                        borderpad=4,
                    )
                fig.update_layout(showlegend=False)
                render_plotly_chart(fig, height=520)
                st.caption(
                    "Dotted lines show the median emissions and median weighted intensity among qualifying routes. "
                    "Bubble size represents voyage count; both axes use log scales to make clustered routes readable."
                )

            with trend_col:
                top_routes = route_summary.head(10)[route_col].astype(str).tolist()
                route_display_func = format_country_pair_option if route_col == "country_pair" else str
                selected_intensity_route = st.selectbox(
                    "Route to inspect",
                    top_routes,
                    format_func=route_display_func,
                    key=f"route_intensity_selection_{route_mode}",
                )
                selected_voyages = df[
                    df[route_col].astype(str).eq(selected_intensity_route)
                    & df["CO2"].notna()
                    & df["delta_dist_km"].notna()
                    & (df["delta_dist_km"] > 0)
                ]
                monthly_intensity = (
                    selected_voyages.groupby("departure_month", dropna=False)
                    .agg(
                        total_CO2=("CO2", "sum"),
                        total_distance_km=("delta_dist_km", "sum"),
                        voyage_count=("CO2", "size"),
                    )
                    .reset_index()
                    .sort_values("departure_month")
                )
                monthly_intensity["weighted_CO2_per_km"] = (
                    monthly_intensity["total_CO2"] / monthly_intensity["total_distance_km"]
                )
                fig = px.line(
                    monthly_intensity,
                    x="departure_month",
                    y="weighted_CO2_per_km",
                    markers=True,
                    title=f"Monthly weighted intensity: {selected_intensity_route}",
                    labels={
                        "departure_month": "Month",
                        "weighted_CO2_per_km": "Distance-weighted CO₂ intensity (kg/km)",
                    },
                    hover_data={"voyage_count": ":,.0f"},
                )
                fig.update_traces(
                    line_color=CLIMATE_COLORS["navy"],
                    hovertemplate=(
                        "<b>%{x}</b><br>Weighted intensity: %{y:.2f} kg/km"
                        "<br>Voyages: %{customdata[0]:,.0f}<extra></extra>"
                    ),
                )
                render_plotly_chart(fig, height=520)

    elif metric_name == "CO2 per DWT-km":
        st.markdown("#### How does vessel size relate to DWT-normalised intensity?")
        minimum_voyages = route_evidence_threshold(route_col)
        st.caption(
            f"Routes require at least {minimum_voyages} voyages with valid distance and DWT. "
            "The metric is weighted as total CO₂ divided by total DWT-km."
        )
        route_analytics = build_route_operational_summary(df, route_col)
        qualified = route_analytics[
            (route_analytics["voyage_count"] >= minimum_voyages)
            & route_analytics["weighted_CO2_per_dwt_km"].notna()
            & route_analytics["mean_dwt"].notna()
        ].copy()

        if qualified.empty:
            st.info("Not enough valid DWT coverage is available for this comparison.")
        else:
            size_col, ship_col = st.columns([1.15, 1])
            with size_col:
                fig = px.scatter(
                    qualified,
                    x="mean_dwt",
                    y="weighted_CO2_per_dwt_km",
                    size="voyage_count",
                    size_max=24,
                    hover_name=route_col,
                    hover_data={
                        "mean_dwt": ":,.0f",
                        "weighted_CO2_per_dwt_km": ":.6f",
                        "total_CO2_million_tonnes": ":.2f",
                        "voyage_count": ":,.0f",
                    },
                    log_x=True,
                    title="Typical vessel size vs normalised intensity",
                    labels={
                        "mean_dwt": "Mean vessel DWT (log scale)",
                        "weighted_CO2_per_dwt_km": "Weighted CO₂ intensity (kg/DWT-km)",
                        "total_CO2_million_tonnes": "Total CO₂ (million tonnes)",
                        "voyage_count": "Voyages",
                    },
                )
                fig.update_traces(marker={"color": CLIMATE_COLORS["blue"], "opacity": 0.72})
                render_plotly_chart(fig, height=500)

            with ship_col:
                top_routes = route_summary.head(10)[route_col].astype(str).tolist()
                route_display_func = format_country_pair_option if route_col == "country_pair" else str
                selected_dwt_route = st.selectbox(
                    "Route to inspect",
                    top_routes,
                    format_func=route_display_func,
                    key=f"route_dwt_selection_{route_mode}",
                )
                selected_route_df = df[df[route_col].astype(str).eq(selected_dwt_route)]
                ship_intensity = aggregate_metric(
                    selected_route_df,
                    "ship_type_clean",
                    "CO2 per DWT-km",
                    12,
                )
                ship_intensity = ship_intensity[
                    ship_intensity["voyage_count"] >= 5
                ].head(6)
                ship_intensity["value_label"] = ship_intensity["CO2_per_dwt_km"].apply(
                    lambda value: f"{value:.6f}"
                )
                ship_plot = ship_intensity.sort_values("CO2_per_dwt_km", ascending=True)
                fig = px.bar(
                    ship_plot,
                    x="CO2_per_dwt_km",
                    y="ship_type_clean",
                    orientation="h",
                    text="value_label",
                    title=f"Leading ship-type intensities: {selected_dwt_route}",
                    labels={
                        "CO2_per_dwt_km": "Weighted CO₂ intensity (kg/DWT-km)",
                        "ship_type_clean": "Ship Type",
                    },
                )
                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    marker_color=CLIMATE_COLORS["teal"],
                    hovertemplate=(
                        "<b>%{y}</b><br>Weighted intensity: %{x:.6f} kg/DWT-km"
                        "<extra></extra>"
                    ),
                )
                ship_axis_max = ship_plot["CO2_per_dwt_km"].max()
                fig.update_xaxes(
                    range=[0, ship_axis_max * 1.45] if ship_axis_max > 0 else None,
                    tickformat=".6f",
                )
                render_plotly_chart(fig, height=500)

    elif metric_name == "Voyage count":
        st.markdown("#### How is activity changing on the busiest routes?")
        st.caption(
            "Select one of the ten busiest routes to compare its monthly traffic and ship-type activity mix."
        )
        top_routes = route_summary.head(10)[route_col].astype(str).tolist()
        route_display_func = format_country_pair_option if route_col == "country_pair" else str
        selected_activity_route = st.selectbox(
            "Route to inspect",
            top_routes,
            format_func=route_display_func,
            key=f"route_activity_selection_{route_mode}",
        )
        selected_route_df = df[df[route_col].astype(str).eq(selected_activity_route)]

        trend_col, mix_col = st.columns([1.1, 1])
        with trend_col:
            monthly_activity = (
                selected_route_df.groupby("departure_month", dropna=False)
                .size()
                .reset_index(name="voyage_count")
                .sort_values("departure_month")
            )
            fig = px.line(
                monthly_activity,
                x="departure_month",
                y="voyage_count",
                markers=True,
                title=f"Monthly voyages: {selected_activity_route}",
                labels={"departure_month": "Month", "voyage_count": "Voyages (records)"},
            )
            fig.update_traces(
                line_color=CLIMATE_COLORS["navy"],
                hovertemplate="<b>%{x}</b><br>Voyages: %{y:,.0f}<extra></extra>",
            )
            render_plotly_chart(fig, height=500)

        with mix_col:
            ship_activity = (
                selected_route_df.groupby("ship_type_clean", dropna=False)
                .size()
                .reset_index(name="voyage_count")
                .sort_values("voyage_count", ascending=False)
            )
            ship_activity["ship_type_clean"] = (
                ship_activity["ship_type_clean"].fillna("Unknown").astype(str)
            )
            leading_ship_types = ship_activity.head(6).copy()
            other_voyages = ship_activity.iloc[6:]["voyage_count"].sum()
            if other_voyages > 0:
                leading_ship_types = pd.concat(
                    [
                        leading_ship_types,
                        pd.DataFrame(
                            {
                                "ship_type_clean": ["Other ship types"],
                                "voyage_count": [other_voyages],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            total_voyages = leading_ship_types["voyage_count"].sum()
            leading_ship_types["share_pct"] = (
                leading_ship_types["voyage_count"] / total_voyages * 100
            )
            leading_ship_types["value_label"] = leading_ship_types.apply(
                lambda row: f"{format_number(row['voyage_count'])} · {row['share_pct']:.1f}%",
                axis=1,
            )
            ship_plot = leading_ship_types.sort_values("voyage_count", ascending=True)
            bar_colors = [
                CLIMATE_COLORS["muted"]
                if ship_type in {"Unknown", "Other ship types"}
                else CLIMATE_COLORS["green"]
                for ship_type in ship_plot["ship_type_clean"]
            ]
            fig = px.bar(
                ship_plot,
                x="voyage_count",
                y="ship_type_clean",
                orientation="h",
                text="value_label",
                custom_data=["share_pct"],
                title=f"Ship-type activity mix: {selected_activity_route}",
                labels={"voyage_count": "Voyages (records)", "ship_type_clean": "Ship Type"},
            )
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{y}</b><br>Voyages: %{x:,.0f}"
                    "<br>Share of route: %{customdata[0]:.1f}%<extra></extra>"
                ),
            )
            add_ship_type_colour_key(fig, CLIMATE_COLORS["green"])
            ship_axis_max = ship_plot["voyage_count"].max()
            fig.update_xaxes(range=[0, ship_axis_max * 1.65] if ship_axis_max > 0 else None)
            render_plotly_chart(fig, height=500, extra_top_margin=34)


# 9. Ship and vessel analysis page

def show_ship_analysis(
    df: pd.DataFrame,
    metric_name: str,
    top_n: int,
) -> None:
    """Render ship type and vessel characteristic analysis."""
    st.subheader("Ship and Vessel Analysis")
    st.caption("Use this view to compare how vessel categories contribute to the selected emissions picture.")

    metric_col = metric_column(metric_name)
    metric_label = format_metric_label(metric_name)
    ship_summary = aggregate_metric(df, "ship_type_clean", metric_name, top_n)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig = px.bar(
            ship_summary.sort_values(metric_col, ascending=True),
            x=metric_col,
            y="ship_type_clean",
            orientation="h",
            title=f"Which ship types rank highest by {metric_label}?",
            labels={metric_col: metric_label, "ship_type_clean": "Ship Type"},
        )
        render_plotly_chart(fig)

    with col2:
        display_dataframe(ship_summary, use_container_width=True, hide_index=True)

    if "vessel_age_group" in df.columns:
        age_order = [
            "0–5 years",
            "6–10 years",
            "11–15 years",
            "16–20 years",
            "21–25 years",
            "26–30 years",
            "31+ years",
            "Unknown / invalid age",
        ]
        age_summary = (
            df.groupby("vessel_age_group", dropna=False, observed=False)
            .agg(
                total_CO2=("CO2", "sum"),
                total_distance_km=("delta_dist_km", "sum"),
                voyage_count=("CO2", "size"),
                unique_vessels=("imo", "nunique"),
            )
            .reset_index()
        )
        age_summary["weighted_CO2_per_km"] = (
            age_summary["total_CO2"]
            / age_summary["total_distance_km"].replace(0, np.nan)
        )
        age_summary["total_CO2_million_tonnes"] = (
            age_summary["total_CO2"] / KG_PER_MILLION_TONNES
        )
        age_summary["vessel_label"] = age_summary["unique_vessels"].apply(
            lambda value: f"{value:,.0f}"
        )
        age_summary["emissions_label"] = age_summary["total_CO2_million_tonnes"].apply(
            lambda value: f"{format_million_tonnes(value)} Mt"
        )

        st.markdown("#### Vessel age profile")
        st.caption(
            "Five-year age bands are used through age 30. Negative or missing ages are separated as "
            "Unknown / invalid age rather than being included in the youngest group."
        )
        age_count_col, age_emissions_col = st.columns(2)

        with age_count_col:
            fig = px.bar(
                age_summary,
                x="vessel_age_group",
                y="unique_vessels",
                text="vessel_label",
                title="How many unique vessels are in each age group?",
                labels={
                    "vessel_age_group": "Vessel Age Group",
                    "unique_vessels": "Unique vessels (IMO)",
                },
                category_orders={"vessel_age_group": age_order},
            )
            age_count_colors = [
                CLIMATE_COLORS["muted"]
                if str(group) == "Unknown / invalid age"
                else CLIMATE_COLORS["navy"]
                for group in age_summary["vessel_age_group"]
            ]
            fig.update_traces(
                marker_color=age_count_colors,
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{x}</b><br>Unique vessels: %{y:,.0f}<extra></extra>"
                ),
            )
            count_axis_max = age_summary["unique_vessels"].max()
            fig.update_yaxes(range=[0, count_axis_max * 1.16] if count_axis_max > 0 else None)
            render_plotly_chart(fig, height=460)

        with age_emissions_col:
            fig = px.bar(
                age_summary,
                x="vessel_age_group",
                y="total_CO2_million_tonnes",
                text="emissions_label",
                title="How much CO₂ is associated with each age group?",
                labels={
                    "vessel_age_group": "Vessel Age Group",
                    "total_CO2_million_tonnes": "Total CO₂ (million tonnes)",
                },
                category_orders={"vessel_age_group": age_order},
            )
            age_emissions_colors = [
                CLIMATE_COLORS["muted"]
                if str(group) == "Unknown / invalid age"
                else CLIMATE_COLORS["teal"]
                for group in age_summary["vessel_age_group"]
            ]
            fig.update_traces(
                marker_color=age_emissions_colors,
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{x}</b><br>Total CO₂: %{y:.2f} million tonnes<extra></extra>"
                ),
            )
            emissions_axis_max = age_summary["total_CO2_million_tonnes"].max()
            fig.update_yaxes(
                range=[0, emissions_axis_max * 1.18] if emissions_axis_max > 0 else None,
                tickformat=".2f",
            )
            render_plotly_chart(fig, height=460)

        with st.expander("Age-group data table"):
            display_dataframe(
                age_summary[
                    [
                        "vessel_age_group",
                        "unique_vessels",
                        "voyage_count",
                        "total_CO2",
                        "weighted_CO2_per_km",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    vessel_detail = (
        df.groupby("ship_type_clean", dropna=False)
        .agg(
            voyage_count=("CO2", "size"),
            total_CO2=("CO2", "sum"),
            mean_CO2=("CO2", "mean"),
            mean_distance_km=("delta_dist_km", "mean"),
            total_distance_km=("delta_dist_km", "sum"),
            mean_dwt=("dwt", "mean"),
            mean_engine_power=("vessel_engine_power", "mean"),
        )
        .reset_index()
        .sort_values("total_CO2", ascending=False)
    )
    vessel_detail["weighted_CO2_per_km"] = (
        vessel_detail["total_CO2"]
        / vessel_detail["total_distance_km"].replace(0, np.nan)
    )

    display_dataframe(vessel_detail, use_container_width=True, hide_index=True)


# 10. Port analysis page

def show_port_analysis(df: pd.DataFrame, top_n: int) -> None:
    """Render port-level analysis and a simple port map."""
    st.subheader("Port Analysis")
    st.caption(
        "Use this view to locate port hotspots within the selected time, country-pair, and vessel scope."
    )

    # Aggregate by port name for ranking so duplicate coordinates do not split the same port.
    origin_all = (
        df.groupby("origin_port_display", dropna=False)
        .agg(
            total_CO2=("CO2", "sum"),
            voyage_count=("CO2", "size"),
            origin_lat=("origin_lat", "first"),
            origin_lon=("origin_lon", "first"),
        )
        .reset_index()
        .sort_values("total_CO2", ascending=False)
    )
    origin_summary = origin_all.head(top_n).copy()

    destination_all = (
        df.groupby("destination_port_display", dropna=False)
        .agg(
            total_CO2=("CO2", "sum"),
            voyage_count=("CO2", "size"),
            destination_lat=("destination_lat", "first"),
            destination_lon=("destination_lon", "first"),
        )
        .reset_index()
        .sort_values("total_CO2", ascending=False)
    )
    destination_summary = destination_all.head(top_n).copy()


    origin_endpoints = origin_all.rename(
        columns={
            "origin_port_display": "port",
            "origin_lat": "lat",
            "origin_lon": "lon",
        }
    )
    destination_endpoints = destination_all.rename(
        columns={
            "destination_port_display": "port",
            "destination_lat": "lat",
            "destination_lon": "lon",
        }
    )
    combined_port_all = (
        pd.concat(
            [
                origin_endpoints[["port", "total_CO2", "voyage_count", "lat", "lon"]],
                destination_endpoints[["port", "total_CO2", "voyage_count", "lat", "lon"]],
            ],
            ignore_index=True,
        )
        .groupby("port", dropna=False)
        .agg(
            total_CO2=("total_CO2", "sum"),
            voyage_count=("voyage_count", "sum"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .reset_index()
        .sort_values("total_CO2", ascending=False)
    )
    combined_port_summary = combined_port_all.head(top_n).copy()
    col1, col2 = st.columns(2)

    with col1:
        origin_plot = origin_summary.sort_values("total_CO2", ascending=False)
        origin_plot = origin_plot.copy()
        origin_plot["total_CO2"] = origin_plot["total_CO2"] / KG_PER_MILLION_TONNES
        fig = px.bar(
            origin_plot,
            x="total_CO2",
            y="origin_port_display",
            orientation="h",
            title=f"Which origin ports are linked to the most CO₂?",
            labels={"total_CO2": "Total CO₂ (million tonnes)", "origin_port_display": "Origin Port"},
            category_orders={
                "origin_port_display": origin_plot["origin_port_display"].tolist()
            },
        )
        render_plotly_chart(fig)

    with col2:
        destination_plot = destination_summary.sort_values("total_CO2", ascending=False)
        destination_plot = destination_plot.copy()
        destination_plot["total_CO2"] = destination_plot["total_CO2"] / KG_PER_MILLION_TONNES
        fig = px.bar(
            destination_plot,
            x="total_CO2",
            y="destination_port_display",
            orientation="h",
            title=f"Which destination ports are linked to the most CO₂?",
            labels={"total_CO2": "Total CO₂ (million tonnes)", "destination_port_display": "Destination Port"},
            category_orders={
                "destination_port_display": destination_plot["destination_port_display"].tolist()
            },
        )
        render_plotly_chart(fig)

    combined_plot = combined_port_summary.sort_values("total_CO2", ascending=False).copy()
    combined_plot["total_CO2"] = combined_plot["total_CO2"] / KG_PER_MILLION_TONNES
    fig = px.bar(
        combined_plot,
        x="total_CO2",
        y="port",
        orientation="h",
        title="Which ports have the highest combined origin and destination CO₂?",
        labels={
            "total_CO2": "Origin + destination CO₂ (million tonnes)",
            "port": "Port",
        },
        category_orders={"port": combined_plot["port"].tolist()},
    )
    render_plotly_chart(fig, height=460)
    st.caption(
        "Combined port CO₂ sums emissions associated with a port as both an origin and a "
        "destination. It is a port-ranking measure; each voyage is represented at both endpoints."
    )

    ports_for_map = combined_port_summary.dropna(subset=["lat", "lon"]).copy()
    ports_for_map = ports_for_map.copy()
    ports_for_map["total_CO2"] = ports_for_map["total_CO2"] / KG_PER_MILLION_TONNES

    if not ports_for_map.empty:
        # The source data is North Sea focused. Filtering the plotted markers as well as zooming the projection prevents stray coordinates from pulling attention away from the European analysis area.
        ports_for_map = ports_for_map.loc[
            ports_for_map["lat"].between(30, 75)
            & ports_for_map["lon"].between(-25, 45)
        ].copy()

    if not ports_for_map.empty:
        emissions = ports_for_map["total_CO2"].clip(lower=0)
        if emissions.max() > emissions.min():
            scaled_emissions = (np.sqrt(emissions) - np.sqrt(emissions.min())) / (
                np.sqrt(emissions.max()) - np.sqrt(emissions.min())
            )
        else:
            scaled_emissions = pd.Series(0.5, index=emissions.index)
        marker_sizes = 12 + (scaled_emissions * 22)

        fig = go.Figure()

        fig.add_trace(
            go.Scattergeo(
                lat=ports_for_map["lat"],
                lon=ports_for_map["lon"],
                mode="markers",
                marker={
                    "size": marker_sizes + 10,
                    "color": "rgba(213, 116, 70, 0.20)",
                    "line": {"width": 0},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lat=ports_for_map["lat"],
                lon=ports_for_map["lon"],
                mode="markers",
                text=ports_for_map["port"],
                customdata=np.column_stack(
                    [ports_for_map["total_CO2"], ports_for_map["voyage_count"]]
                ),
                marker={
                    "size": marker_sizes,
                    "color": "rgba(255, 255, 250, 0.78)",
                    "line": {"color": "#D57446", "width": 2.4},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Total CO₂: %{customdata[0]:.2f} million tonnes<br>"
                    "Voyages: %{customdata[1]:,.0f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.update_geos(
            projection={
                "type": "orthographic",
                "rotation": {"lon": 10, "lat": 52, "roll": 0},
                "scale": 2.65,
            },
            bgcolor="#F5F6F1",
            showframe=False,
            showocean=True,
            oceancolor="#F5F6F1",
            showland=True,
            landcolor="#DCE5D7",
            showlakes=True,
            lakecolor="#F5F6F1",
            showcoastlines=True,
            coastlinecolor="#819184",
            coastlinewidth=0.9,
            showcountries=True,
            countrycolor="#AAB7A9",
            countrywidth=0.7,
        )
        fig.update_layout(
            height=590,
            margin={"l": 12, "r": 12, "t": 72, "b": 24},
            paper_bgcolor="#F5F6F1",
            plot_bgcolor="#F5F6F1",
            title={
                "text": "<b>Where are the main combined port hotspots?</b>",
                "x": 0.02,
                "xanchor": "left",
                "font": {
                    "size": 15,
                    "family": PREMIUM_FONT_STACK,
                    "color": CLIMATE_COLORS["text"],
                },
            },
            hoverlabel={
                "bgcolor": "#F7FAFC",
                "bordercolor": "#D57446",
                "font": {"color": "#172033", "family": PREMIUM_FONT_STACK},
            },
        )
        fig.add_annotation(
            x=0.02,
            y=0.04,
            xref="paper",
            yref="paper",
            text="Larger ring = higher total CO₂",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 12, "color": CLIMATE_COLORS["muted"], "family": PREMIUM_FONT_STACK},
            bgcolor="rgba(245, 246, 241, 0.78)",
            borderpad=5,
        )

        render_plotly_chart(fig, height=590)

    with st.expander("Port ranking data"):
        display_dataframe(combined_port_summary, use_container_width=True, hide_index=True)

    st.markdown("### Nearest-port check")
    st.caption(
        "No port records are merged in this dashboard. Use this diagnostic to identify "
        "nearby port-reference entries for manual review; proximity alone does not establish "
        "that two records represent the same port."
    )

    endpoint_specs = [
        ("port_id_o", "origin_port_display", "origin_lat", "origin_lon"),
        ("port_id_d", "destination_port_display", "destination_lat", "destination_lon"),
    ]
    endpoint_summaries = []
    for id_col, name_col, lat_col, lon_col in endpoint_specs:
        endpoint = df[[id_col, name_col, lat_col, lon_col]].copy()
        endpoint[lat_col] = pd.to_numeric(endpoint[lat_col], errors="coerce")
        endpoint[lon_col] = pd.to_numeric(endpoint[lon_col], errors="coerce")
        endpoint = endpoint.dropna(subset=[lat_col, lon_col])
        endpoint = endpoint.loc[
            endpoint[lat_col].between(-90, 90)
            & endpoint[lon_col].between(-180, 180)
        ]
        endpoint[id_col] = endpoint[id_col].fillna("Missing ID").astype(str)
        endpoint[name_col] = endpoint[name_col].fillna("Unknown port").astype(str)
        endpoint_summary = (
            endpoint.groupby([id_col, name_col], dropna=False)
            .agg(
                lat=(lat_col, "median"),
                lon=(lon_col, "median"),
                endpoint_records=(name_col, "size"),
            )
            .reset_index()
            .rename(columns={id_col: "port_id", name_col: "port"})
        )
        endpoint_summaries.append(endpoint_summary)

    port_reference = (
        pd.concat(endpoint_summaries, ignore_index=True)
        .groupby(["port_id", "port"], dropna=False)
        .agg(
            lat=("lat", "median"),
            lon=("lon", "median"),
            endpoint_records=("endpoint_records", "sum"),
        )
        .reset_index()
    )
    port_reference["selector_label"] = (
        port_reference["port"].astype(str)
        + " ["
        + port_reference["port_id"].astype(str)
        + "]"
    )
    port_reference = port_reference.sort_values(
        ["port", "port_id"], key=lambda values: values.astype(str).str.lower()
    ).reset_index(drop=True)

    if len(port_reference) > 1:
        default_port_index = next(
            (
                index
                for index, label in enumerate(port_reference["selector_label"])
                if "rotterdam" in label.lower()
            ),
            0,
        )
        selected_port_label = st.selectbox(
            "Reference port",
            port_reference["selector_label"].tolist(),
            index=default_port_index,
            help=(
                "Select a port-reference entry to calculate the ten nearest other "
                "entries from their stored coordinates."
            ),
        )
        selected_port = port_reference.loc[
            port_reference["selector_label"].eq(selected_port_label)
        ].iloc[0]

        candidate_ports = port_reference.loc[
            ~port_reference["selector_label"].eq(selected_port_label)
        ].copy()
        earth_radius_km = 6371.0088
        reference_lat = np.radians(float(selected_port["lat"]))
        reference_lon = np.radians(float(selected_port["lon"]))
        candidate_lat = np.radians(candidate_ports["lat"].astype(float))
        candidate_lon = np.radians(candidate_ports["lon"].astype(float))
        delta_lat = candidate_lat - reference_lat
        delta_lon = candidate_lon - reference_lon
        haversine_term = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(reference_lat)
            * np.cos(candidate_lat)
            * np.sin(delta_lon / 2) ** 2
        )
        candidate_ports["distance_km"] = (
            2
            * earth_radius_km
            * np.arcsin(np.sqrt(haversine_term.clip(0, 1)))
        )
        nearest_ports = (
            candidate_ports.nsmallest(10, "distance_km")
            .loc[:, ["port", "port_id", "distance_km", "lat", "lon", "endpoint_records"]]
            .rename(
                columns={
                    "port": "Nearby port",
                    "port_id": "Port ID",
                    "distance_km": "Distance (km)",
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "endpoint_records": "Endpoint records",
                }
            )
        )
        nearest_ports["Distance (km)"] = nearest_ports["Distance (km)"].round(2)
        nearest_ports["Latitude"] = nearest_ports["Latitude"].round(5)
        nearest_ports["Longitude"] = nearest_ports["Longitude"].round(5)
        display_dataframe(nearest_ports, use_container_width=True, hide_index=True)


# 11. Emission intensity page

def show_intensity_analysis(df: pd.DataFrame, metric_name: str) -> None:
    """Render emission intensity analysis."""
    st.subheader("Emission Intensity")
    use_dwt_adjustment = metric_name == "CO2 per DWT-km"
    voyage_metric_col = "CO2_per_dwt_km" if use_dwt_adjustment else "CO2_per_km"
    metric_result_col = (
        "weighted_CO2_per_dwt_km" if use_dwt_adjustment else "weighted_CO2_per_km"
    )
    metric_label = (
        "Capacity-adjusted CO₂ intensity (kg/DWT-km)"
        if use_dwt_adjustment
        else "Distance-weighted CO₂ intensity (kg/km)"
    )
    metric_short_label = "kg/DWT-km" if use_dwt_adjustment else "kg/km"
    metric_format = ".6f" if use_dwt_adjustment else ",.2f"
    st.caption(
        f"Use this view to separate high traffic volume from emissions intensity. "
        f"Current measure: {metric_label}."
    )

    valid_mask = (df["delta_dist_km"] > 0) & (df["CO2"] > 0)
    if use_dwt_adjustment:
        valid_mask &= df["dwt"].notna() & (df["dwt"] > 0)
    valid = df.loc[valid_mask].copy()
    if valid.empty:
        st.warning(
            "No records satisfy the positive CO₂, distance"
            + (" and DWT" if use_dwt_adjustment else "")
            + " requirements under the current filters."
        )
        return

    valid["intensity_denominator"] = valid["delta_dist_km"]
    if use_dwt_adjustment:
        valid["intensity_denominator"] *= valid["dwt"]

    sample_size = min(50_000, len(valid))
    sampled = valid.sample(sample_size, random_state=42) if sample_size > 0 else valid

    fig = px.scatter(
        sampled,
        x="delta_dist_km",
        y="CO2",
        color="ship_type_clean",
        hover_data=[
            "country_o",
            "country_d",
            "origin_port_display",
            "destination_port_display",
            "voyage_scope",
        ],
        log_x=True,
        log_y=True,
        title="How does voyage distance relate to emissions?",
        labels={
            "delta_dist_km": "Distance (km)",
            "CO2": "CO₂ per voyage (kg)",
            "ship_type_clean": "Ship Type",
        },
    )

    fig.update_layout(
        legend={
            "title": {"text": "Ship type"},
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.14,
            "yanchor": "top",
            "font": {"size": 11},
        }
    )
    render_plotly_chart(fig, height=690, extra_bottom_margin=190)

    country_pair_count = valid["country_pair"].dropna().astype(str).nunique()
    port_route_count = valid["port_route_display"].dropna().astype(str).nunique()

    if country_pair_count > 1:
        group_col = "country_pair"
        group_label = "Country Pair"
        min_voyages = 30
        title = f"Which country pairs have the highest CO₂ intensity ({metric_short_label})?"
    elif port_route_count > 1:
        group_col = "port_route_display"
        group_label = "Port-to-port Route"
        min_voyages = 10
        title = f"Which port-to-port routes have the highest CO₂ intensity ({metric_short_label})?"
    else:
        total_co2 = valid["CO2"].sum()
        total_denominator = valid["intensity_denominator"].sum()
        weighted_intensity = total_co2 / total_denominator if total_denominator > 0 else np.nan
        mean_intensity = valid[voyage_metric_col].mean()
        median_intensity = valid[voyage_metric_col].median()

        st.markdown("#### Selected route intensity profile")
        st.caption(
            "A single route is selected, so this section focuses on intensity variation instead of ranking one route against itself."
        )
        render_kpi_cards(
            [
                (metric_label, format(weighted_intensity, metric_format) if pd.notna(weighted_intensity) else "N/A"),
                (f"Mean voyage intensity ({metric_short_label})", format(mean_intensity, metric_format) if pd.notna(mean_intensity) else "N/A"),
                (f"Median voyage intensity ({metric_short_label})", format(median_intensity, metric_format) if pd.notna(median_intensity) else "N/A"),
                ("Voyages (records)", format_number(len(valid))),
            ],
            compact=True,
        )

        ship_intensity = (
            valid.groupby("ship_type_clean", dropna=False)
            .agg(
                voyage_count=("CO2", "size"),
                total_CO2=("CO2", "sum"),
                total_denominator=("intensity_denominator", "sum"),
            )
            .reset_index()
        )
        ship_intensity[metric_result_col] = (
            ship_intensity["total_CO2"] / ship_intensity["total_denominator"]
        )
        ship_intensity = ship_intensity[ship_intensity["voyage_count"] >= 5]
        ship_intensity = ship_intensity.sort_values(metric_result_col, ascending=False)

        monthly_intensity = (
            valid.groupby("departure_month", dropna=False)
            .agg(
                total_CO2=("CO2", "sum"),
                total_denominator=("intensity_denominator", "sum"),
            )
            .reset_index()
            .sort_values("departure_month")
        )
        monthly_intensity[metric_result_col] = (
            monthly_intensity["total_CO2"] / monthly_intensity["total_denominator"]
        )

        col1, col2 = st.columns(2)

        with col1:
            if not ship_intensity.empty:
                fig = px.bar(
                    ship_intensity.sort_values(metric_result_col, ascending=True),
                    x=metric_result_col,
                    y="ship_type_clean",
                    orientation="h",
                    title="Which ship types are most CO₂-intensive on this route?",
                    labels={
                        metric_result_col: metric_label,
                        "ship_type_clean": "Ship Type",
                    },
                )
                render_plotly_chart(fig)
            else:
                st.info("Not enough voyages per ship type to compare intensity reliably.")

        with col2:
            fig = px.line(
                monthly_intensity,
                x="departure_month",
                y=metric_result_col,
                markers=True,
                title="How does weighted CO₂ intensity change over time?",
                labels={
                    "departure_month": "Month",
                    metric_result_col: metric_label,
                },
            )
            render_plotly_chart(fig)

        fig = px.histogram(
            valid,
            x=voyage_metric_col,
            nbins=45,
            color="ship_type_clean",
            title="How widely do voyage-level CO₂ intensities vary?",
            labels={
                voyage_metric_col: f"Voyage CO₂ intensity ({metric_short_label})",
                "ship_type_clean": "Ship Type",
            },
        )
        fig.update_yaxes(title_text="Voyages (records)")
        render_plotly_chart(fig, height=460)

        display_dataframe(ship_intensity, use_container_width=True, hide_index=True)
        return

    route_intensity = (
        valid.groupby(group_col, dropna=False)
        .agg(
            voyage_count=("CO2", "size"),
            total_CO2=("CO2", "sum"),
            total_denominator=("intensity_denominator", "sum"),
        )
        .reset_index()
    )

    route_intensity[metric_result_col] = (
        route_intensity["total_CO2"] / route_intensity["total_denominator"]
    )
    route_intensity = route_intensity[route_intensity["voyage_count"] >= min_voyages]
    route_intensity = route_intensity.sort_values(metric_result_col, ascending=False).head(20)

    if route_intensity.empty:
        st.info("Not enough voyages are available to compare route intensity reliably under the current filters.")
    else:
        fig = px.bar(
            route_intensity.sort_values(metric_result_col, ascending=True),
            x=metric_result_col,
            y=group_col,
            orientation="h",
            title=title,
            labels={metric_result_col: metric_label, group_col: group_label},
        )
        render_plotly_chart(fig)

    display_dataframe(route_intensity, use_container_width=True, hide_index=True)


# 12. Static route map page

def show_od_map(df: pd.DataFrame, top_n: int) -> None:
    """Render a simple origin-destination route map."""
    st.subheader("Route Map")

    map_source = df.dropna(
        subset=["origin_lat", "origin_lon", "destination_lat", "destination_lon"]
    ).copy()
    map_dwt_valid = (
        map_source["CO2"].notna()
        & map_source["delta_dist_km"].notna()
        & (map_source["delta_dist_km"] > 0)
        & map_source["dwt"].notna()
        & (map_source["dwt"] > 0)
    )
    map_source["dwt_valid_CO2"] = map_source["CO2"].where(map_dwt_valid)
    map_source["total_dwt_km"] = (
        map_source["delta_dist_km"] * map_source["dwt"]
    ).where(map_dwt_valid)
    route_map = (
        map_source
        .groupby(
            [
                "port_route_display",
                "origin_port_display",
                "destination_port_display",
                "origin_lat",
                "origin_lon",
                "destination_lat",
                "destination_lon",
            ],
            dropna=False,
        )
        .agg(
            total_CO2=("CO2", "sum"),
            total_distance_km=("delta_dist_km", "sum"),
            voyage_count=("CO2", "size"),
            dwt_valid_CO2=("dwt_valid_CO2", "sum"),
            total_dwt_km=("total_dwt_km", "sum"),
        )
        .reset_index()
    )
    route_map["weighted_CO2_per_km"] = (
        route_map["total_CO2"]
        / route_map["total_distance_km"].replace(0, np.nan)
    )
    route_map["weighted_CO2_per_dwt_km"] = (
        route_map["dwt_valid_CO2"]
        / route_map["total_dwt_km"].replace(0, np.nan)
    )
    route_map = apply_map_metric_style(route_map, "Total CO2").head(top_n)

    render_pydeck_route_map(route_map, f"Top {top_n} Route and Port Activity Map by CO₂")
    display_dataframe(route_map, use_container_width=True, hide_index=True)


# 13. Animated route map page


def show_animated_route_map(df: pd.DataFrame, top_n: int, metric_name: str) -> None:
    """Render a temporal route map with monthly playback."""
    st.subheader("Animated Route Map")

    valid_periods = sorted(df["departure_month"].dropna().astype(str).unique())
    if not valid_periods:
        st.warning("No departure month values are available.")
        return

    metric_config = get_map_metric_config(metric_name)

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        selected_period = st.select_slider(
            "Departure month",
            options=valid_periods,
            value=valid_periods[-1],
        )

    with col2:
        cumulative = st.toggle(
            "Cumulative up to selected month",
            value=False,
        )

    with col3:
        play_animation = st.button("Play monthly animation")

    map_placeholder = st.empty()
    table_placeholder = st.empty()

    def draw_period(period: str) -> None:
        route_summary = build_temporal_route_summary(
            df,
            period=period,
            top_n=top_n,
            cumulative=cumulative,
            metric_name=metric_name,
        )
        title_prefix = "Cumulative route and port activity up to" if cumulative else "Route and port activity in"
        with map_placeholder.container():
            render_pydeck_route_map(route_summary, f"{title_prefix} {period} by {metric_config['label']}")
        with table_placeholder.container():
            display_dataframe(route_summary, use_container_width=True, hide_index=True)

    if play_animation:
        for period in valid_periods:
            draw_period(period)
            time.sleep(0.45)
    else:
        draw_period(selected_period)


# 14. Full-dataset AI assistant


ASSISTANT_GROUPS = {
    "none": None,
    "country_pair": "country_pair",
    "directional_country_route": "route_directional",
    "port_route": "port_route_display",
    "ship_type": "ship_type_clean",
    "vessel_type": "vessel_general_type",
    "vessel_flag": "vessel_flag_2024",
    "departure_year": "departure_year",
    "departure_month": "departure_month",
    "voyage_scope": "voyage_scope",
    "origin_port": "origin_port_display",
    "destination_port": "destination_port_display",
    "vessel_age_group": "vessel_age_group",
    "vessel_size_group": "vessel_size_group_dwt",
}

ASSISTANT_METRICS = {
    "total_co2": ("SUM(COALESCE(CO2, 0))", "total_co2_kg"),
    "voyage_count": ("COUNT(*)", "voyage_count"),
    "vessel_count": ("COUNT(DISTINCT imo)", "vessel_count"),
    "total_distance": ("SUM(COALESCE(delta_dist_km, 0))", "total_distance_km"),
    "weighted_co2_per_km": (
        "SUM(CO2) / NULLIF(SUM(delta_dist_km), 0)",
        "weighted_co2_per_km",
    ),
    "mean_co2_per_km": ("AVG(CO2_per_km)", "mean_co2_per_km"),
    "weighted_co2_per_dwt_km": (
        "SUM(CO2) / NULLIF(SUM(delta_dist_km * dwt), 0)",
        "weighted_co2_per_dwt_km",
    ),
}

ASSISTANT_METRIC_COLUMNS = {
    "total_co2": ["CO2"],
    "voyage_count": [],
    "vessel_count": ["imo"],
    "total_distance": ["delta_dist_km"],
    "weighted_co2_per_km": ["CO2", "delta_dist_km"],
    "mean_co2_per_km": ["CO2_per_km"],
    "weighted_co2_per_dwt_km": ["CO2", "delta_dist_km", "dwt"],
}

ASSISTANT_FILTER_COLUMNS = {
    "years": "departure_year",
    "country_pairs": "country_pair",
    "ship_types": "ship_type_clean",
    "voyage_scopes": "voyage_scope",
    "port_routes": "port_route_display",
    "vessel_types": "vessel_general_type",
}


def get_secret_value(name: str, default: str | None = None) -> str | None:
    """Read an application secret without exposing it in the interface."""
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def build_assistant_tool() -> dict:
    """Define the only analytical operation available to the model."""
    return {
        "type": "function",
        "name": "query_voyage_dataset",
        "description": (
            "Aggregate the approved maritime voyage dataset. Use this tool for every "
            "answer that requires a number, comparison, ranking, or trend."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": list(ASSISTANT_GROUPS),
                    "description": "Use none for a whole-dataset summary.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ASSISTANT_METRICS)},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "years": {"type": "array", "items": {"type": "integer"}},
                        "country_pairs": {"type": "array", "items": {"type": "string"}},
                        "country_codes": {"type": "array", "items": {"type": "string"}},
                        "ship_types": {"type": "array", "items": {"type": "string"}},
                        "voyage_scopes": {"type": "array", "items": {"type": "string"}},
                        "port_routes": {"type": "array", "items": {"type": "string"}},
                        "vessel_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "sort_direction": {"type": "string", "enum": ["desc", "asc"]},
            },
            "required": ["group_by", "metrics"],
            "additionalProperties": False,
        },
        "strict": False,
    }


def normalise_assistant_spec(spec: dict) -> dict:
    """Validate and normalise a model-generated analytical request."""
    group_by = spec.get("group_by", "none")
    if group_by not in ASSISTANT_GROUPS:
        group_by = "none"

    metrics = [metric for metric in spec.get("metrics", []) if metric in ASSISTANT_METRICS]
    if not metrics:
        metrics = ["total_co2"]

    filters = spec.get("filters") or {}
    clean_filters: dict[str, list] = {}
    for key in [*ASSISTANT_FILTER_COLUMNS, "country_codes"]:
        values = filters.get(key, [])
        if isinstance(values, list):
            clean_filters[key] = values[:100]

    raw_quality_rules = spec.get("quality_rules") or {}
    quality_rules = {
        "exclude_same_port": bool(raw_quality_rules.get("exclude_same_port", False)),
        "exclude_generic_ports": bool(
            raw_quality_rules.get("exclude_generic_ports", False)
        ),
        "minimum_distance_km": max(
            0, min(float(raw_quality_rules.get("minimum_distance_km", 0)), 10000)
        ),
    }

    return {
        "group_by": group_by,
        "metrics": metrics[:5],
        "filters": clean_filters,
        "quality_rules": quality_rules,
        "limit": max(1, min(int(spec.get("limit", 10)), 50)),
        "sort_direction": "asc" if spec.get("sort_direction") == "asc" else "desc",
    }


@st.cache_data(show_spinner=False, ttl=3600)
def _execute_full_dataset_query_cached(
    parquet_path: str,
    spec_json: str,
    file_modified_ns: int,
) -> dict:
    """Execute and cache an exact columnar query for an unchanged Parquet file."""
    del file_modified_ns
    if duckdb is None:
        raise RuntimeError("DuckDB is not installed. Run: pip install duckdb")

    path = Path(parquet_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Approved dataset not found: {path}")

    spec = json.loads(spec_json)
    group_column = ASSISTANT_GROUPS[spec["group_by"]]
    metric_parts = [ASSISTANT_METRICS[metric] for metric in spec["metrics"]]

    select_parts: list[str] = []
    if group_column:
        select_parts.append(f"{group_column} AS category")
    select_parts.extend(f"{expression} AS {alias}" for expression, alias in metric_parts)
    if group_column:
        select_parts.append("SUM(COUNT(*)) OVER () AS _matching_rows")
    else:
        select_parts.append("COUNT(*) AS _matching_rows")

    where_parts: list[str] = []
    parameters: list[object] = [str(path)]
    columns_used: set[str] = set()
    if group_column:
        columns_used.add(group_column)
    for metric in spec["metrics"]:
        columns_used.update(ASSISTANT_METRIC_COLUMNS[metric])

    for filter_name, column_name in ASSISTANT_FILTER_COLUMNS.items():
        values = spec["filters"].get(filter_name, [])
        if not values:
            continue
        columns_used.add(column_name)
        placeholders = ", ".join("?" for _ in values)
        where_parts.append(f"{column_name} IN ({placeholders})")
        parameters.extend(values)

    country_codes = [str(value).upper() for value in spec["filters"].get("country_codes", [])]
    if country_codes:
        columns_used.update(["country_o", "country_d"])
        placeholders = ", ".join("?" for _ in country_codes)
        where_parts.append(
            f"(country_o IN ({placeholders}) OR country_d IN ({placeholders}))"
        )
        parameters.extend(country_codes)
        parameters.extend(country_codes)

    quality_rules = spec.get("quality_rules", {})
    if quality_rules.get("exclude_same_port"):
        columns_used.update(["origin_port_display", "destination_port_display"])
        where_parts.extend(
            [
                "origin_port_display IS NOT NULL",
                "destination_port_display IS NOT NULL",
                "origin_port_display <> destination_port_display",
            ]
        )
    minimum_distance_km = float(quality_rules.get("minimum_distance_km", 0))
    if minimum_distance_km > 0:
        columns_used.add("delta_dist_km")
        where_parts.append("delta_dist_km >= ?")
        parameters.append(minimum_distance_km)
    if quality_rules.get("exclude_generic_ports"):
        columns_used.update(["origin_port_display", "destination_port_display"])
        generic_patterns = ["%PILOT%", "%CH16%", "%FOR DESTINATION%", "%CIST%"]
        for pattern in generic_patterns:
            where_parts.append("origin_port_display NOT ILIKE ?")
            parameters.append(pattern)
            where_parts.append("destination_port_display NOT ILIKE ?")
            parameters.append(pattern)

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    group_sql = f"GROUP BY {group_column}" if group_column else ""
    first_alias = metric_parts[0][1]
    order_sql = (
        f"ORDER BY {first_alias} {spec['sort_direction'].upper()} NULLS LAST"
        if group_column
        else ""
    )
    limit_sql = f"LIMIT {spec['limit']}" if group_column else ""

    query = f"""
        SELECT {', '.join(select_parts)}
        FROM read_parquet(?)
        {where_sql}
        {group_sql}
        {order_sql}
        {limit_sql}
    """

    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        result_df = connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()

    if result_df.empty:
        matching_rows = 0
    else:
        matching_rows = int(result_df["_matching_rows"].iloc[0])
    result_df = result_df.drop(columns=["_matching_rows"], errors="ignore")
    result_records = json.loads(result_df.to_json(orient="records", date_format="iso"))
    return {
        "dataset": "North Sea Enriched Voyages",
        "scope": "Exact query over the approved dataset",
        "query_strategy": "Column projection, predicate filtering, and exact aggregation",
        "columns_used": sorted(columns_used),
        "matching_rows": matching_rows,
        "analysis": spec,
        "results": result_records,
    }


def execute_full_dataset_query(parquet_path: str, raw_spec: dict) -> dict:
    """Run a question-specific exact query against the approved full dataset."""
    path = Path(parquet_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Approved dataset not found: {path}")
    spec = normalise_assistant_spec(raw_spec)
    spec_json = json.dumps(spec, sort_keys=True, ensure_ascii=False)
    return _execute_full_dataset_query_cached(
        str(path),
        spec_json,
        path.stat().st_mtime_ns,
    )


CORRIDOR_PRIORITY_QUESTION = "Where should decarbonisation efforts be prioritised?"

CORRIDOR_PRIORITY_CRITERIA = {
    "Emissions impact": "emissions_impact",
    "Emission intensity": "emission_intensity",
    "Activity volume": "activity_volume",
    "Two-way flow stability": "directional_balance",
    "Fleet concentration": "fleet_concentration",
}

CORRIDOR_PRIORITY_BASE_WEIGHTS = {
    "emissions_impact": 0.30,
    "emission_intensity": 0.25,
    "activity_volume": 0.20,
    "directional_balance": 0.15,
    "fleet_concentration": 0.10,
}

DEEP_ANALYSIS_TYPES = {
    "corridor_priority_assessment": (
        "Rank actionable port-to-port corridors using the decision priorities selected "
        "by the user and a consistent minimum-evidence threshold."
    ),
    "route_risk_matrix": (
        "Find corridors that combine high total CO2 with high distance-weighted "
        "CO2 intensity, rather than ranking by only one metric."
    ),
    "emission_growth_drivers": (
        "Identify the country-pair and ship-type combinations that drove the largest "
        "absolute changes between the first and last analysis years."
    ),
    "engine_emission_mix": (
        "Compare main-engine, auxiliary-engine, and boiler emission shares by ship type "
        "and reveal unusually high hotel-load or boiler dependence."
    ),
    "emission_concentration": (
        "Measure how much total CO2 is concentrated among the top 1, 5, and 10 percent "
        "of vessels and country-pair routes."
    ),
    "route_ship_intensity_outliers": (
        "Detect route and ship-type combinations with unusually high weighted CO2 per km "
        "after excluding very short movements and small groups."
    ),
}

PRESET_INVESTIGATIONS = {
    "Which corridors combine high total CO₂ with unusually high intensity?": {
        "analysis_type": "route_risk_matrix"
    },
    "What route and ship-type combinations drove the largest CO₂ growth from 2018 to 2021?": {
        "analysis_type": "emission_growth_drivers",
        "start_year": 2018,
        "end_year": 2021,
    },
    "Where are auxiliary-engine and boiler emissions unusually important?": {
        "analysis_type": "engine_emission_mix"
    },
    "How concentrated are emissions among the top 1%, 5%, and 10% of vessels and routes?": {
        "analysis_type": "emission_concentration"
    },
    "Which route–ship type combinations are intensity outliers after excluding short movements?": {
        "analysis_type": "route_ship_intensity_outliers"
    },
    CORRIDOR_PRIORITY_QUESTION: {
        "analysis_type": "corridor_priority_assessment",
        "priority_criteria": list(CORRIDOR_PRIORITY_BASE_WEIGHTS),
    },
}

ASSISTANT_METHOD_NOTES = {
    "corridor_priority_assessment": (
        "Eligible corridors require at least 500 voyages, 20 vessels, both travel "
        "directions, records in every year from 2018 to 2021, and a median recorded "
        "distance of 100–700 km. Selected criteria are reweighted to total 100%."
    ),
    "route_risk_matrix": (
        "The combined score adds within-dataset percentile ranks for total CO2 and "
        "weighted CO2/km. It is a prioritisation index, not a causal, efficiency, "
        "or regulatory-risk measure. Routes require at least 100 voyages."
    ),
    "emission_growth_drivers": (
        "Absolute changes may reflect activity as well as emissions performance. Only "
        "the two comparison years are used, and groups require at least 20 voyages in "
        "both years."
    ),
    "engine_emission_mix": (
        "Component shares use the sum of main-engine, auxiliary-engine, and boiler CO2. "
        "These component fields do not always reconcile exactly to total CO2."
    ),
    "emission_concentration": (
        "Concentration does not imply intrinsic inefficiency. Traffic volume, distance, "
        "and ship mix also affect total emissions."
    ),
    "route_ship_intensity_outliers": (
        "Movements under 20 km are excluded, and groups require at least 30 voyages and "
        "three vessels. Weighted CO2/km is not adjusted for cargo carried."
    ),
}


def build_deep_analysis_tool() -> dict:
    """Define curated multi-table analyses for questions beyond dashboard charts."""
    return {
        "type": "function",
        "function": {
            "name": "run_deep_analysis",
            "description": (
                "Run a curated exact investigation across several measures. Use this for "
                "hidden patterns, drivers, concentration, component mix, or outliers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": list(DEEP_ANALYSIS_TYPES),
                    },
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"},
                    "country_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "ship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "priority_criteria": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(CORRIDOR_PRIORITY_BASE_WEIGHTS),
                        },
                    },
                },
                "required": ["analysis_type"],
            },
        },
    }


def build_ollama_query_tool() -> dict:
    """Convert the generic query tool to Ollama's function schema."""
    openai_style = build_assistant_tool()
    return {
        "type": "function",
        "function": {
            "name": openai_style["name"],
            "description": openai_style["description"],
            "parameters": openai_style["parameters"],
        },
    }


def normalise_deep_spec(raw_spec: dict) -> dict:
    """Validate a local-model request for a curated analysis."""
    analysis_type = raw_spec.get("analysis_type", "route_risk_matrix")
    if analysis_type not in DEEP_ANALYSIS_TYPES:
        analysis_type = "route_risk_matrix"
    start_year = max(2017, min(int(raw_spec.get("start_year", 2018)), 2021))
    end_year = max(start_year, min(int(raw_spec.get("end_year", 2021)), 2021))
    spec = {
        "analysis_type": analysis_type,
        "country_codes": [
            str(value).upper() for value in raw_spec.get("country_codes", [])[:50]
        ],
        "ship_types": [str(value) for value in raw_spec.get("ship_types", [])[:50]],
    }
    if analysis_type == "emission_growth_drivers":
        spec["start_year"] = start_year
        spec["end_year"] = end_year
    if analysis_type == "corridor_priority_assessment":
        selected_criteria = [
            criterion
            for criterion in raw_spec.get("priority_criteria", [])
            if criterion in CORRIDOR_PRIORITY_BASE_WEIGHTS
        ]
        spec["priority_criteria"] = selected_criteria or list(
            CORRIDOR_PRIORITY_BASE_WEIGHTS
        )
    return spec


def deep_filter_sql(spec: dict, extra_conditions: list[str] | None = None) -> tuple[str, list]:
    """Build safe shared filters for curated DuckDB analyses."""
    conditions = list(extra_conditions or [])
    parameters: list[object] = []
    country_codes = spec.get("country_codes", [])
    if country_codes:
        placeholders = ", ".join("?" for _ in country_codes)
        conditions.append(
            f"(country_o IN ({placeholders}) OR country_d IN ({placeholders}))"
        )
        parameters.extend(country_codes)
        parameters.extend(country_codes)
    ship_types = spec.get("ship_types", [])
    if ship_types:
        placeholders = ", ".join("?" for _ in ship_types)
        conditions.append(f"ship_type_clean IN ({placeholders})")
        parameters.extend(ship_types)
    return (f"WHERE {' AND '.join(conditions)}" if conditions else "", parameters)


@st.cache_data(show_spinner=False, ttl=3600)
def _run_deep_analysis_cached(
    parquet_path: str,
    spec_json: str,
    file_modified_ns: int,
) -> dict:
    """Run a cached exact multi-table investigation against the Parquet dataset."""
    del file_modified_ns
    if duckdb is None:
        raise RuntimeError("DuckDB is not installed. Run: pip install duckdb")
    spec = json.loads(spec_json)
    path = str(Path(parquet_path).expanduser())
    analysis_type = spec["analysis_type"]
    tables: dict[str, list[dict]] = {}
    columns_used: set[str] = set()

    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        if analysis_type == "corridor_priority_assessment":
            selected_criteria = spec["priority_criteria"]
            selected_base_weights = {
                criterion: CORRIDOR_PRIORITY_BASE_WEIGHTS[criterion]
                for criterion in selected_criteria
            }
            weight_total = sum(selected_base_weights.values())
            weights = {
                criterion: weight / weight_total
                for criterion, weight in selected_base_weights.items()
            }
            score_columns = {
                "emissions_impact": "total_co2_percentile",
                "emission_intensity": "intensity_percentile",
                "activity_volume": "activity_percentile",
                "directional_balance": "direction_balance",
                "fleet_concentration": "top_two_ship_type_share_pct / 100.0",
            }
            score_expression = " + ".join(
                f"{weights[criterion]:.12f} * ({score_columns[criterion]})"
                for criterion in selected_criteria
            )
            priority_query = f"""
                WITH base AS (
                    SELECT
                        CASE WHEN origin_port_display < destination_port_display
                            THEN origin_port_display || ' ↔ ' || destination_port_display
                            ELSE destination_port_display || ' ↔ ' || origin_port_display
                        END AS corridor,
                        port_route_display,
                        ship_type_clean,
                        imo,
                        CO2,
                        delta_dist_km,
                        departure_year
                    FROM read_parquet(?)
                    WHERE departure_year BETWEEN 2018 AND 2021
                        AND CO2 IS NOT NULL
                        AND delta_dist_km BETWEEN 20 AND 1500
                        AND origin_port_display IS NOT NULL
                        AND destination_port_display IS NOT NULL
                        AND origin_port_display <> destination_port_display
                        AND country_o IN ('GBR', 'NLD', 'BEL', 'DEU', 'DNK', 'NOR', 'SWE', 'FRA')
                        AND country_d IN ('GBR', 'NLD', 'BEL', 'DEU', 'DNK', 'NOR', 'SWE', 'FRA')
                        AND origin_port_display NOT ILIKE '%PILOT%'
                        AND destination_port_display NOT ILIKE '%PILOT%'
                        AND origin_port_display NOT ILIKE '%CH16%'
                        AND destination_port_display NOT ILIKE '%CH16%'
                ), corridor_stats AS (
                    SELECT
                        corridor,
                        COUNT(*) AS voyage_count,
                        COUNT(DISTINCT imo) AS vessel_count,
                        COUNT(DISTINCT port_route_display) AS direction_count,
                        COUNT(DISTINCT departure_year) AS year_count,
                        SUM(CO2) AS total_co2_kg,
                        SUM(CO2) / NULLIF(SUM(delta_dist_km), 0)
                            AS weighted_co2_per_km,
                        MEDIAN(delta_dist_km) AS median_distance_km
                    FROM base
                    GROUP BY corridor
                    HAVING COUNT(*) >= 500
                        AND COUNT(DISTINCT imo) >= 20
                        AND COUNT(DISTINCT departure_year) = 4
                        AND COUNT(DISTINCT port_route_display) = 2
                        AND MEDIAN(delta_dist_km) BETWEEN 100 AND 700
                ), direction_counts AS (
                    SELECT
                        corridor,
                        MIN(direction_voyages)::DOUBLE / MAX(direction_voyages)
                            AS direction_balance
                    FROM (
                        SELECT corridor, port_route_display, COUNT(*) AS direction_voyages
                        FROM base
                        GROUP BY corridor, port_route_display
                    )
                    GROUP BY corridor
                ), ship_totals AS (
                    SELECT corridor, ship_type_clean, SUM(CO2) AS ship_co2_kg
                    FROM base
                    GROUP BY corridor, ship_type_clean
                ), ship_ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY corridor ORDER BY ship_co2_kg DESC
                        ) AS ship_rank,
                        SUM(ship_co2_kg) OVER (PARTITION BY corridor) AS corridor_co2_kg
                    FROM ship_totals
                ), ship_focus AS (
                    SELECT
                        corridor,
                        100 * SUM(
                            CASE WHEN ship_rank <= 2 THEN ship_co2_kg ELSE 0 END
                        ) / MAX(corridor_co2_kg) AS top_two_ship_type_share_pct
                    FROM ship_ranked
                    GROUP BY corridor
                ), eligible AS (
                    SELECT
                        corridor_stats.*,
                        direction_counts.direction_balance,
                        ship_focus.top_two_ship_type_share_pct,
                        PERCENT_RANK() OVER (ORDER BY total_co2_kg)
                            AS total_co2_percentile,
                        PERCENT_RANK() OVER (ORDER BY weighted_co2_per_km)
                            AS intensity_percentile,
                        PERCENT_RANK() OVER (ORDER BY voyage_count)
                            AS activity_percentile
                    FROM corridor_stats
                    JOIN direction_counts USING (corridor)
                    JOIN ship_focus USING (corridor)
                )
                SELECT
                    *,
                    {score_expression} AS priority_score
                FROM eligible
                ORDER BY priority_score DESC, total_co2_kg DESC
                LIMIT 10
            """
            priority_frame = connection.execute(priority_query, [path]).fetchdf()
            tables["Decarbonisation priority corridors"] = json.loads(
                priority_frame.to_json(orient="records")
            )
            spec["criterion_weights"] = weights
            columns_used.update(
                [
                    "departure_year",
                    "country_o",
                    "country_d",
                    "origin_port_display",
                    "destination_port_display",
                    "port_route_display",
                    "ship_type_clean",
                    "imo",
                    "CO2",
                    "delta_dist_km",
                ]
            )

        elif analysis_type == "unity_governance_use_case":
            candidate_query = """
                WITH base AS (
                    SELECT
                        CASE WHEN origin_port_display < destination_port_display
                            THEN origin_port_display || ' ↔ ' || destination_port_display
                            ELSE destination_port_display || ' ↔ ' || origin_port_display
                        END AS corridor,
                        port_route_display,
                        ship_type_clean,
                        imo,
                        CO2,
                        delta_dist_km,
                        departure_year
                    FROM read_parquet(?)
                    WHERE departure_year BETWEEN 2018 AND 2021
                        AND CO2 IS NOT NULL
                        AND delta_dist_km BETWEEN 20 AND 1500
                        AND origin_port_display IS NOT NULL
                        AND destination_port_display IS NOT NULL
                        AND origin_port_display <> destination_port_display
                        AND country_o IN ('GBR', 'NLD', 'BEL', 'DEU', 'DNK', 'NOR', 'SWE', 'FRA')
                        AND country_d IN ('GBR', 'NLD', 'BEL', 'DEU', 'DNK', 'NOR', 'SWE', 'FRA')
                        AND origin_port_display NOT ILIKE '%PILOT%'
                        AND destination_port_display NOT ILIKE '%PILOT%'
                        AND origin_port_display NOT ILIKE '%CH16%'
                        AND destination_port_display NOT ILIKE '%CH16%'
                ), corridor_stats AS (
                    SELECT
                        corridor,
                        COUNT(*) AS voyage_count,
                        COUNT(DISTINCT imo) AS vessel_count,
                        COUNT(DISTINCT port_route_display) AS direction_count,
                        COUNT(DISTINCT ship_type_clean) AS ship_type_count,
                        COUNT(DISTINCT departure_year) AS year_count,
                        SUM(CO2) AS total_co2_kg,
                        SUM(CO2) / NULLIF(SUM(delta_dist_km), 0) AS weighted_co2_per_km,
                        MEDIAN(delta_dist_km) AS median_distance_km
                    FROM base
                    GROUP BY corridor
                    HAVING COUNT(*) >= 500
                        AND COUNT(DISTINCT imo) >= 20
                        AND COUNT(DISTINCT departure_year) = 4
                        AND COUNT(DISTINCT port_route_display) = 2
                        AND MEDIAN(delta_dist_km) BETWEEN 100 AND 700
                ), direction_counts AS (
                    SELECT corridor, MIN(direction_voyages)::DOUBLE / MAX(direction_voyages)
                        AS direction_balance
                    FROM (
                        SELECT corridor, port_route_display, COUNT(*) AS direction_voyages
                        FROM base
                        GROUP BY corridor, port_route_display
                    )
                    GROUP BY corridor
                ), ship_totals AS (
                    SELECT corridor, ship_type_clean, SUM(CO2) AS ship_co2_kg
                    FROM base
                    GROUP BY corridor, ship_type_clean
                ), ship_ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY corridor ORDER BY ship_co2_kg DESC
                        ) AS ship_rank,
                        SUM(ship_co2_kg) OVER (PARTITION BY corridor) AS corridor_co2_kg
                    FROM ship_totals
                ), ship_focus AS (
                    SELECT corridor,
                        100 * SUM(CASE WHEN ship_rank <= 2 THEN ship_co2_kg ELSE 0 END)
                            / MAX(corridor_co2_kg) AS top_two_ship_type_share_pct
                    FROM ship_ranked
                    GROUP BY corridor
                ), eligible AS (
                    SELECT
                        corridor_stats.*,
                        direction_counts.direction_balance,
                        ship_focus.top_two_ship_type_share_pct,
                        PERCENT_RANK() OVER (ORDER BY total_co2_kg) AS total_co2_percentile,
                        PERCENT_RANK() OVER (
                            ORDER BY weighted_co2_per_km
                        ) AS intensity_percentile,
                        PERCENT_RANK() OVER (ORDER BY voyage_count) AS activity_percentile,
                        PERCENT_RANK() OVER (ORDER BY vessel_count) AS vessel_percentile
                    FROM corridor_stats
                    JOIN direction_counts USING (corridor)
                    JOIN ship_focus USING (corridor)
                )
                SELECT *,
                    0.25 * total_co2_percentile
                    + 0.20 * intensity_percentile
                    + 0.15 * activity_percentile
                    + 0.05 * vessel_percentile
                    + 0.15 * direction_balance
                    + 0.10 * (top_two_ship_type_share_pct / 100.0)
                    + 0.10 * CASE
                        WHEN median_distance_km BETWEEN 120 AND 400 THEN 1.0
                        ELSE 0.7
                    END AS use_case_score
                FROM eligible
                ORDER BY use_case_score DESC
                LIMIT 10
            """
            shortlist = connection.execute(candidate_query, [path]).fetchdf()
            tables["Evidence-based 3D corridor shortlist"] = json.loads(
                shortlist.to_json(orient="records")
            )
            if shortlist.empty:
                selected_corridor = None
            else:
                selected_corridor = str(shortlist.iloc[0]["corridor"])

            if selected_corridor:
                corridor_expression = """
                    CASE WHEN origin_port_display < destination_port_display
                        THEN origin_port_display || ' ↔ ' || destination_port_display
                        ELSE destination_port_display || ' ↔ ' || origin_port_display
                    END
                """
                shared_scope = f"""
                    {corridor_expression} = ?
                    AND departure_year BETWEEN 2018 AND 2021
                    AND delta_dist_km BETWEEN 20 AND 1500
                    AND CO2 IS NOT NULL
                """
                annual_query = f"""
                    SELECT
                        departure_year,
                        COUNT(*) AS voyage_count,
                        COUNT(DISTINCT imo) AS vessel_count,
                        SUM(CO2) AS total_co2_kg,
                        SUM(delta_dist_km) AS total_distance_km,
                        SUM(CO2) / NULLIF(SUM(delta_dist_km), 0)
                            AS weighted_co2_per_km,
                        MEDIAN(delta_dist_km) AS median_distance_km
                    FROM read_parquet(?)
                    WHERE {shared_scope}
                    GROUP BY departure_year
                    ORDER BY departure_year
                """
                annual = connection.execute(
                    annual_query, [path, selected_corridor]
                ).fetchdf()
                tables["Selected corridor annual profile"] = json.loads(
                    annual.to_json(orient="records")
                )

                ship_query = f"""
                    SELECT
                        ship_type_clean,
                        COUNT(*) AS voyage_count,
                        COUNT(DISTINCT imo) AS vessel_count,
                        SUM(CO2) AS total_co2_kg,
                        100 * SUM(CO2) / SUM(SUM(CO2)) OVER () AS co2_share_pct,
                        SUM(CO2) / NULLIF(SUM(delta_dist_km), 0)
                            AS weighted_co2_per_km,
                        100 * SUM(
                            COALESCE(CO2_AE_kg, 0) + COALESCE(CO2_boiler_kg, 0)
                        ) / NULLIF(SUM(
                            COALESCE(CO2_ME_kg, 0) + COALESCE(CO2_AE_kg, 0)
                            + COALESCE(CO2_boiler_kg, 0)
                        ), 0) AS non_propulsion_share_pct
                    FROM read_parquet(?)
                    WHERE {shared_scope}
                    GROUP BY ship_type_clean
                    ORDER BY total_co2_kg DESC
                    LIMIT 10
                """
                ship_profile = connection.execute(
                    ship_query, [path, selected_corridor]
                ).fetchdf()
                tables["Selected corridor ship-type profile"] = json.loads(
                    ship_profile.to_json(orient="records")
                )

                vessel_query = f"""
                    WITH vessel_stats AS (
                        SELECT
                            imo,
                            vessel_name,
                            ship_type_clean,
                            COUNT(*) AS voyage_count,
                            SUM(CO2) AS total_co2_kg,
                            SUM(CO2) / NULLIF(SUM(delta_dist_km), 0)
                                AS weighted_co2_per_km
                        FROM read_parquet(?)
                        WHERE {shared_scope}
                        GROUP BY imo, vessel_name, ship_type_clean
                    )
                    SELECT *,
                        100 * total_co2_kg / SUM(total_co2_kg) OVER () AS co2_share_pct
                    FROM vessel_stats
                    ORDER BY total_co2_kg DESC
                    LIMIT 10
                """
                vessel_profile = connection.execute(
                    vessel_query, [path, selected_corridor]
                ).fetchdf()
                tables["Selected corridor vessel profile"] = json.loads(
                    vessel_profile.to_json(orient="records")
                )

                direction_query = f"""
                    SELECT
                        port_route_display,
                        COUNT(*) AS voyage_count,
                        COUNT(DISTINCT imo) AS vessel_count,
                        SUM(CO2) AS total_co2_kg,
                        SUM(CO2) / NULLIF(SUM(delta_dist_km), 0)
                            AS weighted_co2_per_km
                    FROM read_parquet(?)
                    WHERE {shared_scope}
                    GROUP BY port_route_display
                    ORDER BY voyage_count DESC
                """
                direction_profile = connection.execute(
                    direction_query, [path, selected_corridor]
                ).fetchdf()
                tables["Selected corridor directional profile"] = json.loads(
                    direction_profile.to_json(orient="records")
                )

            columns_used.update(
                [
                    "departure_year",
                    "country_o",
                    "country_d",
                    "origin_port_display",
                    "destination_port_display",
                    "port_route_display",
                    "ship_type_clean",
                    "imo",
                    "vessel_name",
                    "CO2",
                    "delta_dist_km",
                    "CO2_ME_kg",
                    "CO2_AE_kg",
                    "CO2_boiler_kg",
                ]
            )

        elif analysis_type == "route_risk_matrix":
            where_sql, filter_params = deep_filter_sql(
                spec,
                ["CO2 IS NOT NULL", "delta_dist_km > 0"],
            )
            query = f"""
                WITH route_stats AS (
                    SELECT
                        country_pair,
                        COUNT(*) AS voyage_count,
                        COUNT(DISTINCT imo) AS vessel_count,
                        SUM(CO2) AS total_co2_kg,
                        SUM(CO2) / NULLIF(SUM(delta_dist_km), 0) AS weighted_co2_per_km
                    FROM read_parquet(?)
                    {where_sql}
                    GROUP BY country_pair
                    HAVING COUNT(*) >= 100
                ), scored AS (
                    SELECT *,
                        PERCENT_RANK() OVER (ORDER BY total_co2_kg) AS total_co2_percentile,
                        PERCENT_RANK() OVER (ORDER BY weighted_co2_per_km) AS intensity_percentile
                    FROM route_stats
                )
                SELECT *,
                    total_co2_percentile + intensity_percentile AS combined_risk_score
                FROM scored
                WHERE total_co2_percentile >= 0.75 OR intensity_percentile >= 0.75
                ORDER BY combined_risk_score DESC, total_co2_kg DESC
                LIMIT 25
            """
            frame = connection.execute(query, [path, *filter_params]).fetchdf()
            tables["High-volume and high-intensity corridors"] = json.loads(
                frame.to_json(orient="records")
            )
            columns_used.update(["country_pair", "imo", "CO2", "delta_dist_km"])

        elif analysis_type == "emission_growth_drivers":
            where_sql, filter_params = deep_filter_sql(
                spec,
                ["departure_year IN (?, ?)", "CO2 IS NOT NULL"],
            )
            query = f"""
                WITH drivers AS (
                    SELECT
                        country_pair,
                        ship_type_clean,
                        SUM(CASE WHEN departure_year = ? THEN CO2 ELSE 0 END) AS start_co2_kg,
                        SUM(CASE WHEN departure_year = ? THEN CO2 ELSE 0 END) AS end_co2_kg,
                        COUNT(CASE WHEN departure_year = ? THEN 1 END) AS start_voyages,
                        COUNT(CASE WHEN departure_year = ? THEN 1 END) AS end_voyages
                    FROM read_parquet(?)
                    {where_sql}
                    GROUP BY country_pair, ship_type_clean
                )
                SELECT *,
                    end_co2_kg - start_co2_kg AS absolute_change_kg,
                    100 * (end_co2_kg - start_co2_kg) / NULLIF(start_co2_kg, 0) AS percent_change
                FROM drivers
                WHERE start_voyages >= 20 AND end_voyages >= 20
                ORDER BY absolute_change_kg DESC
                LIMIT 25
            """
            parameters = [
                spec["start_year"],
                spec["end_year"],
                spec["start_year"],
                spec["end_year"],
                path,
                spec["start_year"],
                spec["end_year"],
                *filter_params,
            ]
            frame = connection.execute(query, parameters).fetchdf()
            tables[
                f"Largest emission growth drivers, {spec['start_year']} to {spec['end_year']}"
            ] = json.loads(frame.to_json(orient="records"))
            columns_used.update(
                ["departure_year", "country_pair", "ship_type_clean", "CO2"]
            )

        elif analysis_type == "engine_emission_mix":
            where_sql, filter_params = deep_filter_sql(spec, ["CO2 > 0"])
            query = f"""
                SELECT
                    ship_type_clean,
                    COUNT(*) AS voyage_count,
                    SUM(CO2) AS total_co2_kg,
                    100 * SUM(COALESCE(CO2_ME_kg, 0))
                        / NULLIF(SUM(COALESCE(CO2_ME_kg, 0) + COALESCE(CO2_AE_kg, 0)
                        + COALESCE(CO2_boiler_kg, 0)), 0) AS main_engine_share_pct,
                    100 * SUM(COALESCE(CO2_AE_kg, 0))
                        / NULLIF(SUM(COALESCE(CO2_ME_kg, 0) + COALESCE(CO2_AE_kg, 0)
                        + COALESCE(CO2_boiler_kg, 0)), 0) AS auxiliary_engine_share_pct,
                    100 * SUM(COALESCE(CO2_boiler_kg, 0))
                        / NULLIF(SUM(COALESCE(CO2_ME_kg, 0) + COALESCE(CO2_AE_kg, 0)
                        + COALESCE(CO2_boiler_kg, 0)), 0) AS boiler_share_pct,
                    100 * (SUM(COALESCE(CO2_AE_kg, 0)) + SUM(COALESCE(CO2_boiler_kg, 0)))
                        / NULLIF(SUM(COALESCE(CO2_ME_kg, 0) + COALESCE(CO2_AE_kg, 0)
                        + COALESCE(CO2_boiler_kg, 0)), 0) AS non_propulsion_share_pct
                FROM read_parquet(?)
                {where_sql}
                GROUP BY ship_type_clean
                HAVING COUNT(*) >= 100
                ORDER BY non_propulsion_share_pct DESC
                LIMIT 25
            """
            frame = connection.execute(query, [path, *filter_params]).fetchdf()
            tables["Ship types with high non-propulsion emission shares"] = json.loads(
                frame.to_json(orient="records")
            )
            columns_used.update(
                ["ship_type_clean", "CO2", "CO2_ME_kg", "CO2_AE_kg", "CO2_boiler_kg"]
            )

        elif analysis_type == "emission_concentration":
            where_sql, filter_params = deep_filter_sql(spec, ["CO2 IS NOT NULL"])
            entity_queries = {
                "Vessel concentration": "imo",
                "Country-pair concentration": "country_pair",
            }
            for table_name, entity_column in entity_queries.items():
                query = f"""
                    WITH entity_stats AS (
                        SELECT {entity_column} AS entity, SUM(CO2) AS total_co2_kg
                        FROM read_parquet(?)
                        {where_sql}
                        GROUP BY {entity_column}
                    ), ranked AS (
                        SELECT *,
                            ROW_NUMBER() OVER (ORDER BY total_co2_kg DESC) AS rank_number,
                            COUNT(*) OVER () AS entity_count
                        FROM entity_stats
                    )
                    SELECT
                        MAX(entity_count) AS unique_entities,
                        SUM(total_co2_kg) AS total_co2_kg,
                        100 * SUM(CASE WHEN rank_number <= CEIL(entity_count * 0.01)
                            THEN total_co2_kg ELSE 0 END) / SUM(total_co2_kg) AS top_1pct_share,
                        100 * SUM(CASE WHEN rank_number <= CEIL(entity_count * 0.05)
                            THEN total_co2_kg ELSE 0 END) / SUM(total_co2_kg) AS top_5pct_share,
                        100 * SUM(CASE WHEN rank_number <= CEIL(entity_count * 0.10)
                            THEN total_co2_kg ELSE 0 END) / SUM(total_co2_kg) AS top_10pct_share
                    FROM ranked
                """
                frame = connection.execute(query, [path, *filter_params]).fetchdf()
                tables[table_name] = json.loads(frame.to_json(orient="records"))
                columns_used.update([entity_column, "CO2"])

        elif analysis_type == "route_ship_intensity_outliers":
            where_sql, filter_params = deep_filter_sql(
                spec,
                ["CO2 IS NOT NULL", "delta_dist_km >= 20"],
            )
            query = f"""
                SELECT
                    country_pair,
                    port_route_display,
                    ship_type_clean,
                    COUNT(*) AS voyage_count,
                    COUNT(DISTINCT imo) AS vessel_count,
                    SUM(CO2) / NULLIF(SUM(delta_dist_km), 0) AS weighted_co2_per_km,
                    QUANTILE_CONT(CO2_per_km, 0.95) AS p95_voyage_co2_per_km,
                    SUM(CO2) AS total_co2_kg
                FROM read_parquet(?)
                {where_sql}
                GROUP BY country_pair, port_route_display, ship_type_clean
                HAVING COUNT(*) >= 30 AND COUNT(DISTINCT imo) >= 3
                ORDER BY weighted_co2_per_km DESC
                LIMIT 25
            """
            frame = connection.execute(query, [path, *filter_params]).fetchdf()
            tables["Route and ship-type intensity outliers"] = json.loads(
                frame.to_json(orient="records")
            )
            columns_used.update(
                [
                    "country_pair",
                    "port_route_display",
                    "ship_type_clean",
                    "imo",
                    "CO2",
                    "delta_dist_km",
                    "CO2_per_km",
                ]
            )

        date_range = connection.execute(
            "SELECT MIN(departure_year), MAX(departure_year) FROM read_parquet(?)",
            [path],
        ).fetchone()
        metadata = pq.ParquetFile(path).metadata
        matching_rows = int(metadata.num_rows)
    finally:
        connection.close()

    return {
        "dataset": "North Sea Enriched Voyages",
        "scope": "Exact curated investigation over the approved dataset",
        "analysis": spec,
        "analysis_description": DEEP_ANALYSIS_TYPES[analysis_type],
        "query_strategy": "Question-specific column projection and exact aggregation",
        "columns_used": sorted(columns_used),
        "matching_rows": matching_rows,
        "dataset_departure_year_range": [int(date_range[0]), int(date_range[1])],
        "tables": tables,
    }


def run_deep_analysis(parquet_path: str, raw_spec: dict) -> dict:
    """Run a curated deep analysis using the unchanged full dataset."""
    path = Path(parquet_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Approved dataset not found: {path}")
    spec = normalise_deep_spec(raw_spec)
    return _run_deep_analysis_cached(
        str(path),
        json.dumps(spec, sort_keys=True, ensure_ascii=False),
        path.stat().st_mtime_ns,
    )


def assistant_instructions() -> str:
    """Return the dataset-specific instructions supplied to the local model."""
    country_reference = ", ".join(f"{code}={name}" for code, name in COUNTRY_NAME_MAP.items())
    return f"""
You are the data assistant for the North Sea Shipping Emissions Explorer.
Every numerical claim must come from one of the provided data tools. Never estimate values.
For hidden patterns, drivers, concentration, component mix, or outliers, prefer
run_deep_analysis over a simple ranking query.
The tool creates a question-specific query that reads only required columns and matching
records from the approved Parquet dataset. Its aggregations are exact and never sampled.
Use country_pair for undirected pairs, route_directional for directional country routes,
and port_route for port-to-port movements. Country pairs use sorted ISO3 labels such as
GBR ↔ NLD. Map common country names to these ISO3 codes using: {country_reference}.
When a standalone question says only "route" without naming a route level, interpret it
as a directional port-to-port route. The application will exclude same-port movements,
very short movements, and generic pilot or temporary location labels. Use country_pair
only when the user explicitly asks for a country pair or corridor.
When the current question explicitly refers to an earlier result using phrases such as
"this route", "the first one", "that ship type", or "what about it", inherit the relevant
verified entity, filters, and time scope from the recent conversation. Do not carry prior
filters into a new standalone question that does not refer back to them.
Use weighted_co2_per_km for aggregate route intensity unless the user explicitly asks
for the mean of voyage-level intensity. Prefer questions that compare multiple measures,
explain a pattern, or support a decision over shallow single-metric rankings.
If the question cannot be answered with the available dimensions or metrics, say so.
""".strip()


def get_ollama_model() -> str:
    """Return the locally hosted model selected for the assistant."""
    return get_secret_value("OLLAMA_MODEL", "qwen3:4b-instruct") or "qwen3:4b-instruct"


def get_ollama_client() -> object:
    """Create a client for the local Ollama service."""
    if OllamaClient is None:
        raise RuntimeError("The Ollama Python package is not installed.")
    host = get_secret_value("OLLAMA_HOST", "http://localhost:11434")
    return OllamaClient(host=host)


def ollama_readiness() -> tuple[bool, str]:
    """Check whether the local service and selected model are available."""
    if OllamaClient is None:
        return False, "Install the Ollama Python package with `pip install ollama`."
    model = get_ollama_model()
    try:
        response = get_ollama_client().list()
        model_items = getattr(response, "models", [])
        available_models = {
            getattr(item, "model", None)
            or (item.get("model") if isinstance(item, dict) else None)
            for item in model_items
        }
    except Exception:
        return False, "Start the Ollama application on this Mac."
    if model not in available_models:
        return False, f"Download the local model with `ollama pull {model}`."
    return True, f"Local model ready: {model}"


def question_country_codes(question: str) -> list[str]:
    """Extract known country names and ISO3 codes from a user question."""
    lowered = question.lower()
    aliases = {
        "uk": "GBR",
        "britain": "GBR",
        "great britain": "GBR",
        "dutch": "NLD",
    }
    found: set[str] = set()
    for alias, code in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            found.add(code)
    for code, country_name in COUNTRY_NAME_MAP.items():
        if re.search(rf"\b{re.escape(code.lower())}\b", lowered):
            found.add(code)
        if country_name.lower() in lowered:
            found.add(code)
    return sorted(found)


def apply_question_guardrails(
    question: str,
    tool_name: str,
    raw_arguments: dict,
) -> tuple[str, dict]:
    """Enforce explicit question constraints after local-model planning."""
    lowered = question.lower()
    deep_type = None
    if "concentrat" in lowered and ("vessel" in lowered or "route" in lowered):
        deep_type = "emission_concentration"
    elif ("auxiliary" in lowered or "boiler" in lowered) and "emission" in lowered:
        deep_type = "engine_emission_mix"
    elif "outlier" in lowered and ("route" in lowered or "ship" in lowered):
        deep_type = "route_ship_intensity_outliers"
    elif (
        "corridor" in lowered
        and "high" in lowered
        and "intensity" in lowered
        and ("total" in lowered or "volume" in lowered)
    ):
        deep_type = "route_risk_matrix"
    elif (
        ("growth" in lowered or "drove" in lowered or "driver" in lowered)
        and "route" in lowered
        and "ship" in lowered
    ):
        deep_type = "emission_growth_drivers"

    years = sorted(
        {
            int(value)
            for value in re.findall(r"\b(20(?:17|18|19|20|21))\b", question)
        }
    )
    country_codes = question_country_codes(question)

    if deep_type:
        deep_arguments: dict = {"analysis_type": deep_type}
        if deep_type == "emission_growth_drivers" and years:
            deep_arguments["start_year"] = years[0]
            deep_arguments["end_year"] = years[-1]
        if country_codes:
            deep_arguments["country_codes"] = country_codes
        return "run_deep_analysis", deep_arguments

    arguments = dict(raw_arguments)
    filters = dict(arguments.get("filters") or {})
    if years:
        filters["years"] = years
    if country_codes:
        if len(country_codes) == 2 and ("between" in lowered or "pair" in lowered):
            filters["country_pairs"] = [" ↔ ".join(sorted(country_codes))]
            filters["country_codes"] = []
        else:
            filters["country_codes"] = country_codes
    arguments["filters"] = filters

    asks_for_route = bool(re.search(r"\broutes?\b", lowered))
    if "ship type" in lowered:
        arguments["group_by"] = "ship_type"
    elif "country pair" in lowered:
        arguments["group_by"] = "country_pair"
    elif "corridor" in lowered:
        arguments["group_by"] = "country_pair"
    elif asks_for_route:
        arguments["group_by"] = "port_route"
        arguments["quality_rules"] = {
            "exclude_same_port": True,
            "exclude_generic_ports": True,
            "minimum_distance_km": 20,
        }
    elif "vessel age" in lowered or "age group" in lowered:
        arguments["group_by"] = "vessel_age_group"
    elif "by year" in lowered or "annual" in lowered or "yearly" in lowered:
        arguments["group_by"] = "departure_year"
    elif "by month" in lowered or "monthly" in lowered:
        arguments["group_by"] = "departure_month"

    if "voyage count" in lowered or "number of voyages" in lowered:
        arguments["metrics"] = ["voyage_count"]
    elif "vessel count" in lowered or "number of vessels" in lowered:
        arguments["metrics"] = ["vessel_count"]
    elif "dwt" in lowered and ("intensity" in lowered or "per" in lowered):
        arguments["metrics"] = ["weighted_co2_per_dwt_km"]
    elif "per km" in lowered or "intensity" in lowered:
        arguments["metrics"] = ["weighted_co2_per_km"]
    elif "distance" in lowered and "co2" not in lowered and "co₂" not in lowered:
        arguments["metrics"] = ["total_distance"]
    elif "co2" in lowered or "co₂" in lowered or "emission" in lowered:
        arguments["metrics"] = ["total_co2"]

    top_match = re.search(r"\btop\s+(\d{1,2})\b", lowered)
    if top_match:
        arguments["limit"] = min(int(top_match.group(1)), 50)
    if any(word in lowered for word in ["highest", "most", "largest", "top"]):
        arguments["sort_direction"] = "desc"
    return "query_voyage_dataset", arguments


def formatted_evidence_value(column: str, value: object) -> str:
    """Format deterministic evidence values for a concise narrative."""
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value) if isinstance(value, (int, float, np.number)) else None
    if numeric_value is None:
        return str(value)
    if "pct" in column or "percentile" in column:
        return f"{numeric_value:.2f}%" if "pct" in column else f"{numeric_value:.3f}"
    if "per_km" in column:
        return f"{numeric_value:,.2f} kg/km"
    if "per_dwt" in column:
        return f"{numeric_value:.6f} kg/DWT-km"
    if "co2" in column and ("kg" in column or column.endswith("co2")):
        return f"{format_million_tonnes(kg_to_million_tonnes(numeric_value))} million tonnes"
    if "count" in column or "voyage" in column or "vessel" in column or "entities" in column:
        return f"{numeric_value:,.0f}"
    return f"{numeric_value:,.2f}"


def build_verified_answer(evidence: dict) -> str:
    """Create a structured deterministic report from verified query output."""
    analysis = evidence.get("analysis", {})
    analysis_type = analysis.get("analysis_type")
    tables = evidence.get("tables", {})

    if analysis_type == "corridor_priority_assessment":
        rows = tables.get("Decarbonisation priority corridors", [])
        if not rows:
            return "No port-to-port corridors met the minimum evidence threshold."
        leader = rows[0]
        criterion_label_by_key = {
            value: label for label, value in CORRIDOR_PRIORITY_CRITERIA.items()
        }
        weights = analysis.get("criterion_weights", {})
        priority_lines = [
            f"- {criterion_label_by_key.get(criterion, criterion)}: {weight * 100:.0f}%"
            for criterion, weight in weights.items()
        ]
        evidence_lines = [
            f"- **{row['corridor']}**: score {row['priority_score']:.3f}; "
            f"{format_million_tonnes(kg_to_million_tonnes(row['total_co2_kg']))} million tonnes CO₂; "
            f"{row['weighted_co2_per_km']:,.2f} kg/km; "
            f"{row['voyage_count']:,.0f} voyages; "
            f"{row['direction_balance'] * 100:.1f}% two-way balance; "
            f"{row['top_two_ship_type_share_pct']:.1f}% in its two leading ship types."
            for row in rows[:5]
        ]
        return (
            "**Direct answer**  \n"
            f"**{leader['corridor']}** is the highest-priority port-to-port corridor "
            "under the selected decision priorities. Its combined priority score is "
            f"{leader['priority_score']:.3f} among corridors that meet the minimum "
            "evidence threshold.\n\n"
            "**Decision priorities used**  \n"
            + "  \n".join(priority_lines)
            + "\n\n"
            "**Key evidence**  \n"
            + "  \n".join(evidence_lines)
            + "\n\n"
            "**Why it matters**  \n"
            "This result identifies a corridor where a targeted programme could combine "
            "meaningful emissions coverage with a sufficiently active, repeatable and "
            "operationally focused traffic pattern. Changing the selected priorities may "
            "change the recommended corridor, so the result should be treated as a "
            "transparent decision-support ranking rather than a fixed policy conclusion."
        )

    if analysis_type == "unity_governance_use_case":
        shortlist = tables.get("Evidence-based 3D corridor shortlist", [])
        annual = tables.get("Selected corridor annual profile", [])
        ships = tables.get("Selected corridor ship-type profile", [])
        vessels = tables.get("Selected corridor vessel profile", [])
        if not shortlist:
            return "No corridor met the minimum evidence and visual-suitability thresholds."
        selected = shortlist[0]
        runner_up = shortlist[1] if len(shortlist) > 1 else None
        first_year = annual[0] if annual else {}
        last_year = annual[-1] if annual else {}
        leading_ships = ships[:2]
        leading_vessels = vessels[:2]
        ship_share = sum(row.get("co2_share_pct", 0) for row in leading_ships)
        vessel_share = sum(row.get("co2_share_pct", 0) for row in leading_vessels)
        ship_lines = [
            f"- {row['ship_type_clean']}: {row['co2_share_pct']:.2f}% of corridor CO₂, "
            f"{row['voyage_count']:,.0f} voyages and "
            f"{row['weighted_co2_per_km']:,.2f} kg/km."
            for row in leading_ships
        ]
        vessel_lines = [
            f"- {row.get('vessel_name') or 'IMO ' + str(row['imo'])}: "
            f"{row['co2_share_pct']:.2f}% of corridor CO₂ across "
            f"{row['voyage_count']:,.0f} recorded voyages."
            for row in leading_vessels
        ]
        runner_up_text = (
            f" The runner-up is {runner_up['corridor']} with a score of "
            f"{runner_up['use_case_score']:.3f}."
            if runner_up
            else ""
        )
        return (
            "**Recommended Unity 3D use case**  \n"
            f"Focus on **{selected['corridor']}**. It ranks first among eligible North Sea "
            f"port corridors with a multi-criteria suitability score of "
            f"{selected['use_case_score']:.3f}.{runner_up_text}\n\n"
            "**Why this corridor is analytically stronger than a simple 'highest CO₂' choice**  \n"
            f"- It represents {format_million_tonnes(kg_to_million_tonnes(selected['total_co2_kg']))} million tonnes of CO₂ across "
            f"{selected['voyage_count']:,.0f} voyages and {selected['vessel_count']:,.0f} "
            f"vessels.  \n"
            f"- Its weighted intensity is {selected['weighted_co2_per_km']:,.2f} kg/km, "
            f"with a typical recorded movement of {selected['median_distance_km']:,.0f} km.  \n"
            f"- Both directions are well represented: the smaller direction has "
            f"{selected['direction_balance'] * 100:.1f}% as many voyages as the larger one.  \n"
            f"- The two leading ship types account for {ship_share:.2f}% of corridor CO₂, "
            "giving the 3D story a clear operational focus.  \n"
            "- It has qualifying data in every year from 2018 to 2021 and avoids short "
            "port-only movements and generic pilot-area labels.\n\n"
            "**The emissions story inside the corridor**  \n"
            + "  \n".join(ship_lines)
            + "\n\n"
            f"The two leading vessels alone account for {vessel_share:.2f}% of corridor CO₂:  \n"
            + "  \n".join(vessel_lines)
            + "\n\n"
            f"Recorded activity changed from {first_year.get('voyage_count', 0):,.0f} voyages "
            f"and {format_million_tonnes(kg_to_million_tonnes(first_year.get('total_co2_kg', 0)))} million tonnes CO₂ in "
            f"{first_year.get('departure_year', 2018)} to "
            f"{last_year.get('voyage_count', 0):,.0f} voyages and "
            f"{format_million_tonnes(kg_to_million_tonnes(last_year.get('total_co2_kg', 0)))} million tonnes CO₂ in "
            f"{last_year.get('departure_year', 2021)}. Weighted intensity fell from "
            f"{first_year.get('weighted_co2_per_km', 0):,.2f} to "
            f"{last_year.get('weighted_co2_per_km', 0):,.2f} kg/km. This is a pattern to "
            "investigate, not proof of an efficiency improvement.\n\n"
            "**Data-supported governance direction**  \n"
            "1. Prioritise a bilateral UK–Netherlands corridor programme because the route is "
            "regular, bidirectional and operationally concentrated.  \n"
            "2. Target the small recurring Ro-Ro and passenger fleet first; vessel-specific "
            "fuel transition, retrofit and operational trials can cover a large emission share.  \n"
            "3. Treat propulsion fuel, speed and scheduling as the primary analytical levers, "
            "then evaluate shore power as a complementary port measure.  \n"
            "4. Use the 3D scene to compare direction, year, ship type and individual vessel, "
            "so the visualisation supports monitoring rather than acting as decoration.\n\n"
            "**How the selection algorithm works**  \n"
            "Eligible corridors must have at least 500 voyages, 20 vessels, both directions, "
            "four complete analysis years and a median movement of 100–700 km. The score "
            "combines total CO₂ (25%), weighted CO₂/km (20%), activity (15%), directional "
            "balance (15%), ship-type focus (10%), map-suitable distance (10%) and vessel "
            "coverage (5%).\n\n"
            "**Important limitation before making policy claims**  \n"
            "This is a prioritisation framework, not a causal model. The voyage emissions are "
            "modelled, port matching should be visually checked, and 2024 vessel particulars "
            "may not represent ownership or flag status during 2018–2021. High-intensity "
            "passenger-vessel records should also be validated before the corridor is used for "
            "formal governance recommendations."
        )

    if analysis_type == "emission_concentration":
        vessel = tables.get("Vessel concentration", [{}])[0]
        route = tables.get("Country-pair concentration", [{}])[0]
        top_ten_gap = route.get("top_10pct_share", 0) - vessel.get("top_10pct_share", 0)
        route_entities = int(route.get("unique_entities", 0))
        vessel_entities = int(vessel.get("unique_entities", 0))
        return (
            "**Direct answer**  \n"
            f"Emissions are more concentrated across country-pair routes than across vessels. "
            f"The top 10% of country pairs account for {route.get('top_10pct_share', 0):.2f}% "
            f"of CO₂, compared with {vessel.get('top_10pct_share', 0):.2f}% for vessels, "
            f"a gap of {top_ten_gap:.2f} percentage points.\n\n"
            "**Key evidence**  \n"
            f"- The calculation covers {vessel_entities:,} vessels and "
            f"{route_entities:,} country-pair routes.  \n"
            f"- Top 1%: vessels {vessel.get('top_1pct_share', 0):.2f}%; "
            f"country pairs {route.get('top_1pct_share', 0):.2f}%.  \n"
            f"- Top 5%: vessels {vessel.get('top_5pct_share', 0):.2f}%; "
            f"country pairs {route.get('top_5pct_share', 0):.2f}%.  \n"
            f"- Top 10%: vessels {vessel.get('top_10pct_share', 0):.2f}%; "
            f"country pairs {route.get('top_10pct_share', 0):.2f}%.\n\n"
            "**Why it matters**  \n"
            "The route network has a pronounced corridor structure: a relatively small set "
            "of country connections carries most recorded emissions. This suggests that "
            "corridor-level interventions could cover a large share of emissions, while a "
            "vessel-level strategy would need to address a broader population."
        )

    if analysis_type == "emission_growth_drivers":
        rows = next(iter(tables.values()), [])
        if not rows:
            return "No route and ship-type groups met the minimum activity threshold."
        leader = rows[0]
        evidence_lines = [
            f"- {row['country_pair']} · {row['ship_type_clean']}: "
            f"{format_million_tonnes(kg_to_million_tonnes(row['start_co2_kg']))} → "
            f"{format_million_tonnes(kg_to_million_tonnes(row['end_co2_kg']))} million tonnes; "
            f"change {format_million_tonnes(kg_to_million_tonnes(row['absolute_change_kg']))} million tonnes "
            f"({row['percent_change']:.1f}%), with voyages "
            f"{row['start_voyages']:,.0f} → {row['end_voyages']:,.0f}."
            for row in rows[:5]
        ]
        voyage_change = leader["end_voyages"] - leader["start_voyages"]
        return (
            "**Direct answer**  \n"
            f"The largest absolute increase was {leader['country_pair']} · "
            f"{leader['ship_type_clean']}, rising by "
            f"{format_million_tonnes(kg_to_million_tonnes(leader['absolute_change_kg']))} million tonnes between "
            f"{analysis['start_year']} and {analysis['end_year']}. Its voyage count changed "
            f"by {voyage_change:+,.0f} over the same comparison years.\n\n"
            "**Key evidence**  \n" + "  \n".join(evidence_lines) + "\n\n"
            "**Why it matters**  \n"
            "The ranking identifies where the dataset's net increase is concentrated. A large "
            "absolute rise accompanied by more voyages points toward activity growth; a rise "
            "without similar voyage growth may justify checking distance, vessel size and "
            "emission intensity. These are diagnostic signals rather than causal conclusions."
        )

    if analysis_type == "engine_emission_mix":
        rows = next(iter(tables.values()), [])
        if not rows:
            return "No ship types met the minimum activity threshold."
        leader = rows[0]
        evidence_lines = [
            f"- {row['ship_type_clean']}: {row['non_propulsion_share_pct']:.2f}% "
            f"non-propulsion share ({row['auxiliary_engine_share_pct']:.2f}% auxiliary, "
            f"{row['boiler_share_pct']:.2f}% boiler), across "
            f"{row['voyage_count']:,.0f} voyages."
            for row in rows[:5]
        ]
        second_share = rows[1]["non_propulsion_share_pct"] if len(rows) > 1 else 0
        return (
            "**Direct answer**  \n"
            f"{leader['ship_type_clean']} has the highest combined auxiliary-engine and boiler "
            f"share at {leader['non_propulsion_share_pct']:.2f}%. This is "
            f"{leader['non_propulsion_share_pct'] - second_share:.2f} percentage points above "
            "the second-ranked ship type.\n\n"
            "**Key evidence**  \n" + "  \n".join(evidence_lines) + "\n\n"
            "**Why it matters**  \n"
            "A high non-propulsion share means auxiliary systems and boilers form an unusually "
            "large part of the reported engine-component emissions. Possible operational "
            "explanations include hotel loads, onboard services, heating demand or time spent "
            "at low propulsion load, but these causes are not directly observed here."
        )

    if analysis_type == "route_risk_matrix":
        rows = next(iter(tables.values()), [])
        if not rows:
            return "No corridors met the minimum activity and risk thresholds."
        leader = rows[0]
        evidence_lines = [
            f"- {row['country_pair']}: {format_million_tonnes(kg_to_million_tonnes(row['total_co2_kg']))} million tonnes total CO₂, "
            f"{row['weighted_co2_per_km']:,.2f} kg/km, combined score "
            f"{row['combined_risk_score']:.3f}, {row['voyage_count']:,.0f} voyages and "
            f"{row['vessel_count']:,.0f} vessels."
            for row in rows[:5]
        ]
        return (
            "**Direct answer**  \n"
            f"{leader['country_pair']} ranks highest when total CO₂ and weighted CO₂/km are "
            f"considered together. It is at the {leader['total_co2_percentile'] * 100:.1f}th "
            f"percentile for total CO₂ and the {leader['intensity_percentile'] * 100:.1f}th "
            "percentile for weighted intensity within qualifying corridors.\n\n"
            "**Key evidence**  \n" + "  \n".join(evidence_lines) + "\n\n"
            "**Why it matters**  \n"
            "These corridors matter for two reasons at once: they contribute substantial "
            "absolute emissions and are also emission-intensive per kilometre. They are more "
            "useful candidates for detailed investigation than routes selected on volume alone."
        )

    if analysis_type == "route_ship_intensity_outliers":
        rows = next(iter(tables.values()), [])
        if not rows:
            return "No route and ship-type combinations met the outlier thresholds."
        leader = rows[0]
        evidence_lines = [
            f"- {row['port_route_display']} · {row['ship_type_clean']}: "
            f"{row['weighted_co2_per_km']:,.2f} kg/km; voyage-level 95th percentile "
            f"{row['p95_voyage_co2_per_km']:,.2f} kg/km; {row['voyage_count']:,.0f} voyages "
            f"and {row['vessel_count']:,.0f} vessels."
            for row in rows[:5]
        ]
        return (
            "**Direct answer**  \n"
            f"{leader['port_route_display']} · {leader['ship_type_clean']} has the highest "
            f"weighted intensity among qualifying groups at "
            f"{leader['weighted_co2_per_km']:,.2f} kg/km, based on "
            f"{leader['voyage_count']:,.0f} voyages by {leader['vessel_count']:,.0f} vessels.\n\n"
            "**Key evidence**  \n" + "  \n".join(evidence_lines) + "\n\n"
            "**Why it matters**  \n"
            "These combinations remain unusually emission-intensive after removing very short "
            "movements and tiny groups. That makes them stronger candidates for operational "
            "review than a raw voyage-level ranking, although vessel size and cargo utilisation "
            "may still explain part of the pattern."
        )

    rows = evidence.get("results", [])
    if not rows:
        return "No records matched the interpreted question."
    metrics = analysis.get("metrics", [])
    aliases = [ASSISTANT_METRICS[metric][1] for metric in metrics]
    group_by = analysis.get("group_by", "none").replace("_", " ")
    if analysis.get("group_by") == "none":
        values = [
            f"- {alias.replace('_', ' ')}: {formatted_evidence_value(alias, rows[0].get(alias))}"
            for alias in aliases
        ]
        return (
            "**Direct answer**  \nThe requested full-scope metrics are shown below.\n\n"
            "**Key evidence**  \n" + "  \n".join(values) + "\n\n"
            "**Why it matters**  \nAggregate values describe the selected data scope; they do not by "
            "themselves separate changes in activity from changes in emission efficiency."
        )
    first_alias = aliases[0]
    evidence_lines = [
        f"- {row.get('category', 'Unknown')}: "
        f"{formatted_evidence_value(first_alias, row.get(first_alias))}"
        for row in rows
    ]
    group_label = {
        "country_pair": "country-pair corridors",
        "directional_country_route": "directional country routes",
        "port_route": "directional port-to-port routes",
        "ship_type": "ship types",
        "vessel_type": "vessel types",
        "vessel_flag": "vessel flags",
        "departure_year": "years",
        "departure_month": "months",
        "voyage_scope": "voyage scopes",
        "origin_port": "origin ports",
        "destination_port": "destination ports",
        "vessel_age_group": "vessel age groups",
        "vessel_size_group": "vessel size groups",
    }.get(analysis.get("group_by"), f"{group_by} groups")
    metric_label = {
        "total_co2_kg": "total CO2",
        "voyage_count": "voyage count",
        "vessel_count": "vessel count",
        "total_distance_km": "total distance",
        "weighted_co2_per_km": "weighted CO2 per km",
        "mean_co2_per_km": "mean CO2 per km",
        "weighted_co2_per_dwt_km": "weighted CO2 per DWT-km",
    }.get(first_alias, first_alias.replace("_", " "))
    ranked_categories = ", ".join(
        str(row.get("category", "Unknown")) for row in rows
    )
    if len(evidence_lines) > 1:
        direct_answer = (
            f"The top {len(evidence_lines)} {group_label} by {metric_label}, in descending "
            f"order, are {ranked_categories}."
        )
    else:
        direct_answer = (
            f"{rows[0].get('category', 'Unknown')} ranks first by {metric_label} among "
            f"the returned {group_label}."
        )
    return (
        "**Direct answer**  \n"
        f"{direct_answer}\n\n"
        f"**Key evidence: top {len(evidence_lines)} verified results**  \n"
        + "  \n".join(evidence_lines)
        + "\n\n"
        "**Why it matters**  \nA high aggregate ranking can be driven by activity volume, distance, "
        "vessel characteristics or emission intensity. Use a normalised metric when the "
        "question concerns efficiency rather than total contribution."
    )


def build_assistant_history_context(history: list[dict]) -> str:
    """Build compact verified context so follow-up questions retain analytical scope."""
    context_parts: list[str] = []
    for message in history[-6:]:
        role = message.get("role", "user").title()
        content = message.get("content", "")[:1400]
        context_parts.append(f"{role}: {content}")
        evidence = message.get("evidence") or {}
        if not evidence:
            continue
        verified_context: dict[str, object] = {
            "analysis": evidence.get("analysis", {}),
        }
        if evidence.get("results"):
            verified_context["top_results"] = evidence["results"][:3]
        elif evidence.get("tables"):
            first_table_name, first_table_rows = next(
                iter(evidence["tables"].items()), ("", [])
            )
            verified_context["table"] = first_table_name
            verified_context["top_results"] = first_table_rows[:3]
        context_parts.append(
            "Verified previous analysis: "
            + json.dumps(verified_context, ensure_ascii=False, default=str)[:2200]
        )
    return "\n".join(context_parts)


def apply_followup_guardrails(
    question: str,
    history: list[dict],
    tool_name: str,
    raw_arguments: dict,
) -> tuple[str, dict]:
    """Resolve explicit follow-up references from the last verified analysis."""
    lowered = question.lower()
    reference_phrases = (
        "this route",
        "that route",
        "same route",
        "this corridor",
        "that corridor",
        "same corridor",
        "first corridor",
        "top corridor",
        "first one",
        "top one",
        "this ship type",
        "that ship type",
        "same ship type",
        "same period",
        "same years",
    )
    if not any(phrase in lowered for phrase in reference_phrases):
        return tool_name, raw_arguments

    previous_evidence = next(
        (
            message.get("evidence")
            for message in reversed(history)
            if message.get("role") == "assistant" and message.get("evidence")
        ),
        None,
    )
    if not previous_evidence or tool_name != "query_voyage_dataset":
        return tool_name, raw_arguments

    arguments = dict(raw_arguments)
    filters = dict(arguments.get("filters") or {})
    previous_analysis = previous_evidence.get("analysis", {})
    previous_filters = previous_analysis.get("filters", {})
    for filter_name, values in previous_filters.items():
        if values and not filters.get(filter_name):
            filters[filter_name] = values

    top_result: dict = {}
    if previous_evidence.get("results"):
        top_result = previous_evidence["results"][0]
    elif previous_evidence.get("tables"):
        _, first_table_rows = next(iter(previous_evidence["tables"].items()), ("", []))
        if first_table_rows:
            top_result = first_table_rows[0]

    if top_result.get("country_pair") and not filters.get("country_pairs"):
        filters["country_pairs"] = [top_result["country_pair"]]
    elif top_result.get("port_route_display") and not filters.get("port_routes"):
        filters["port_routes"] = [top_result["port_route_display"]]
    elif top_result.get("ship_type_clean") and not filters.get("ship_types"):
        filters["ship_types"] = [top_result["ship_type_clean"]]
    elif top_result.get("category"):
        previous_group = previous_analysis.get("group_by")
        category_filter_map = {
            "country_pair": "country_pairs",
            "port_route": "port_routes",
            "ship_type": "ship_types",
            "vessel_type": "vessel_types",
        }
        filter_name = category_filter_map.get(previous_group)
        if filter_name and not filters.get(filter_name):
            filters[filter_name] = [top_result["category"]]

    arguments["filters"] = filters
    return tool_name, arguments


def ask_local_dataset(
    question: str,
    parquet_path: str,
    history: list[dict],
) -> tuple[str, dict]:
    """Use a local model to plan, execute, and explain a verified data analysis."""
    preset_spec = PRESET_INVESTIGATIONS.get(question.strip())
    if preset_spec:
        evidence = run_deep_analysis(parquet_path, preset_spec)
        return build_verified_answer(evidence), evidence

    client = get_ollama_client()
    model = get_ollama_model()
    prior_context = build_assistant_history_context(history)
    question_with_context = (
        f"Recent conversation:\n{prior_context}\n\nCurrent question: {question}"
        if prior_context
        else question
    )

    messages = [
        {"role": "system", "content": assistant_instructions()},
        {"role": "user", "content": question_with_context},
    ]
    planning_response = client.chat(
        model=model,
        messages=messages,
        tools=[build_ollama_query_tool(), build_deep_analysis_tool()],
        think=False,
        options={"temperature": 0},
    )
    tool_calls = getattr(planning_response.message, "tool_calls", None) or []
    if not tool_calls:
        raise RuntimeError("The assistant could not create a supported data query.")

    tool_call = tool_calls[0]
    tool_name = tool_call.function.name
    tool_arguments = tool_call.function.arguments
    if isinstance(tool_arguments, str):
        tool_arguments = json.loads(tool_arguments)
    tool_name, tool_arguments = apply_question_guardrails(
        question,
        tool_name,
        tool_arguments,
    )
    tool_name, tool_arguments = apply_followup_guardrails(
        question,
        history,
        tool_name,
        tool_arguments,
    )
    if tool_name == "run_deep_analysis":
        evidence = run_deep_analysis(parquet_path, tool_arguments)
    elif tool_name == "query_voyage_dataset":
        evidence = execute_full_dataset_query(parquet_path, tool_arguments)
    else:
        raise RuntimeError("The local model selected an unsupported data tool.")
    return build_verified_answer(evidence), evidence


def friendly_assistant_error(exc: Exception) -> str:
    """Convert local-model and configuration failures into concise user guidance."""
    error_text = str(exc)
    lower_error = error_text.lower()
    if "connection" in lower_error or "refused" in lower_error:
        return "The local Ollama service is not running. Open Ollama, then try again."
    if "not found" in lower_error and "model" in lower_error:
        return f"The local model is not installed. Run `ollama pull {get_ollama_model()}`."
    if "ollama python package" in lower_error:
        return "The local AI package is missing. Run `pip install ollama`."
    return "The assistant could not complete that request. Please try a more specific question."


def render_assistant_evidence(evidence: dict) -> None:
    """Show a compact audit trail for a generated answer."""
    analysis = evidence.get("analysis", {})
    if evidence.get("tables"):
        st.caption(
            f"Exact investigation over the approved dataset containing "
            f"{evidence.get('matching_rows', 0):,} voyage records."
        )
    else:
        st.caption(
            f"Calculated from {evidence.get('matching_rows', 0):,} matching voyage records "
            "in the full approved dataset."
        )
    if analysis.get("analysis_type"):
        investigation_name = analysis["analysis_type"].replace("_", " ").title()
        st.markdown(f"**Investigation:** {investigation_name}")
        if evidence.get("analysis_description"):
            st.caption(evidence["analysis_description"])
    else:
        st.markdown(
            f"**Grouped by:** {analysis.get('group_by', 'none').replace('_', ' ')}  \n"
            f"**Metrics:** {', '.join(analysis.get('metrics', [])).replace('_', ' ')}"
        )
    quality_rules = analysis.get("quality_rules", {})
    active_quality_rules = {
        key: value
        for key, value in quality_rules.items()
        if value not in (False, 0, 0.0, None)
    }
    if active_quality_rules:
        route_definition_parts = []
        if active_quality_rules.get("exclude_same_port"):
            route_definition_parts.append("same-port movements excluded")
        if active_quality_rules.get("exclude_generic_ports"):
            route_definition_parts.append("generic location labels excluded")
        minimum_distance = active_quality_rules.get("minimum_distance_km")
        if minimum_distance:
            route_definition_parts.append(
                f"movements under {minimum_distance:g} km excluded"
            )
        st.markdown(
            "**Route definition:** Directional port-to-port routes; "
            + "; ".join(route_definition_parts)
            + "."
        )
    analysis_type = analysis.get("analysis_type")
    method_note = ASSISTANT_METHOD_NOTES.get(analysis_type)
    if not method_note and not evidence.get("tables"):
        method_note = (
            "Values are exact aggregates over the matching voyage records. Rankings by "
            "total CO2 describe contribution, not efficiency; use a normalised intensity "
            "metric for efficiency comparisons."
        )
    if method_note:
        st.markdown(f"**Method note:** {method_note}")
    columns_used = evidence.get("columns_used", [])
    if columns_used:
        st.markdown(f"**Columns read:** {', '.join(columns_used)}")
    filters = analysis.get("filters", {})
    active_filters = {key: value for key, value in filters.items() if value}
    if active_filters:
        st.markdown(f"**Filters:** `{json.dumps(active_filters, ensure_ascii=False)}`")
    results = evidence.get("results", [])
    if results:
        display_dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    for table_name, table_rows in evidence.get("tables", {}).items():
        st.markdown(f"**{table_name}**")
        if table_rows:
            display_dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )


@st.fragment
def render_data_assistant(df: pd.DataFrame) -> None:
    """Render the floating full-dataset assistant without changing dashboard layout."""
    if "data_assistant_messages" not in st.session_state:
        st.session_state.data_assistant_messages = []

    with st.container(key="data_assistant_launcher"):
        with st.popover("Ask the data", icon=":material/auto_awesome:"):
            st.markdown(
                """
                <div class="assistant-header">
                    <div class="assistant-mark">AI</div>
                    <div>
                        <div class="assistant-title">Maritime Data Assistant</div>
                        <div class="assistant-subtitle">Private local AI · exact DuckDB analysis</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            dataset_name = st.selectbox(
                "Dataset",
                list(APPROVED_DATASETS),
                disabled=len(APPROVED_DATASETS) == 1,
                key="assistant_dataset",
            )
            dataset_config = APPROVED_DATASETS[dataset_name]
            minimum_year = int(df["departure_year"].min())
            maximum_year = int(df["departure_year"].max())
            st.markdown(
                f"""
                <div class="assistant-dataset-card">
                    <div class="assistant-dataset-row">
                        <div class="assistant-dataset-name">{html.escape(dataset_name)}</div>
                        <div class="assistant-ready">Local</div>
                    </div>
                    <div class="assistant-dataset-meta">
                        {len(df):,} voyage records &nbsp;·&nbsp; {minimum_year}–{maximum_year}<br>
                        {html.escape(dataset_config['description'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            local_ready, _ = ollama_readiness()
            if not local_ready:
                st.warning(
                    "The assistant is temporarily unavailable.",
                    icon=":material/info:",
                )

            messages = st.session_state.data_assistant_messages
            internal_design_terms = (
                "unity 3d governance use case",
                "recommended unity 3d use case",
            )
            if any(
                any(term in message.get("content", "").lower() for term in internal_design_terms)
                for message in messages
            ):
                messages = [
                    message
                    for message in messages
                    if not any(
                        term in message.get("content", "").lower()
                        for term in internal_design_terms
                    )
                ]
                st.session_state.data_assistant_messages = messages
            for message in messages:
                if "insufficient_quota" in message.get("content", ""):
                    message["content"] = (
                        "The assistant now uses the free local model. Clear this old OpenAI "
                        "message and ask the question again."
                    )
            if messages:
                message_area = st.container(height=480, border=False)
                with message_area:
                    for message in messages:
                        avatar = (
                            ":material/auto_awesome:"
                            if message["role"] == "assistant"
                            else ":material/person:"
                        )
                        with st.chat_message(message["role"], avatar=avatar):
                            st.markdown(message["content"])
            else:
                st.markdown('<div class="assistant-section-label">Try asking</div>', unsafe_allow_html=True)

            suggestions = list(PRESET_INVESTIGATIONS)
            suggested_question = None
            priority_spec = None
            if not messages:
                for index, suggestion in enumerate(suggestions):
                    if st.button(
                        suggestion,
                        key=f"assistant_suggestion_{index}",
                        use_container_width=True,
                        disabled=not local_ready,
                    ):
                        if suggestion == CORRIDOR_PRIORITY_QUESTION:
                            st.session_state.assistant_priority_builder_open = True
                        else:
                            st.session_state.assistant_priority_builder_open = False
                            suggested_question = suggestion

                if st.session_state.get("assistant_priority_builder_open", False):
                    with st.container(key="assistant_priority_builder"):
                        st.markdown(
                            """
                            <div class="assistant-priority-title">Set decision priorities</div>
                            <div class="assistant-priority-copy">
                                Choose what should influence the corridor ranking. The selected
                                priorities are reweighted automatically to total 100%.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.form("assistant_priority_form"):
                            selected_priority_labels = st.multiselect(
                                "Decision priorities",
                                list(CORRIDOR_PRIORITY_CRITERIA),
                                default=list(CORRIDOR_PRIORITY_CRITERIA),
                                help=(
                                    "Emissions impact uses total CO₂; emission intensity uses "
                                    "distance-weighted CO₂/km; activity uses voyage count; "
                                    "two-way stability compares both directions; fleet "
                                    "concentration measures the share in the two leading ship types."
                                ),
                            )
                            priority_submitted = st.form_submit_button(
                                "Run priority assessment",
                                icon=":material/analytics:",
                                use_container_width=True,
                            )
                        if priority_submitted:
                            if selected_priority_labels:
                                priority_spec = {
                                    "analysis_type": "corridor_priority_assessment",
                                    "priority_criteria": [
                                        CORRIDOR_PRIORITY_CRITERIA[label]
                                        for label in selected_priority_labels
                                    ],
                                }
                                suggested_question = CORRIDOR_PRIORITY_QUESTION
                                st.session_state.assistant_priority_builder_open = False
                            else:
                                st.warning(
                                    "Select at least one decision priority.",
                                    icon=":material/info:",
                                )

            with st.form("data_assistant_form", clear_on_submit=True):
                question = st.text_input(
                    "Ask a question",
                    placeholder="Ask for a pattern, explanation, decision, or governance insight...",
                    label_visibility="collapsed",
                    disabled=not local_ready,
                )
                submitted = st.form_submit_button(
                    "Ask",
                    icon=":material/arrow_upward:",
                    use_container_width=True,
                    disabled=not local_ready,
                )

            prompt = suggested_question or (question.strip() if submitted else "")
            if prompt:
                previous_history = list(messages)
                st.session_state.data_assistant_messages.append(
                    {"role": "user", "content": prompt}
                )
                try:
                    with st.spinner("Analysing the full dataset..."):
                        if priority_spec:
                            evidence = run_deep_analysis(
                                dataset_config["path"], priority_spec
                            )
                            answer = build_verified_answer(evidence)
                        else:
                            answer, evidence = ask_local_dataset(
                                prompt,
                                dataset_config["path"],
                                previous_history,
                            )
                    st.session_state.data_assistant_messages.append(
                        {"role": "assistant", "content": answer, "evidence": evidence}
                    )
                except Exception as exc:
                    st.session_state.data_assistant_messages.append(
                        {
                            "role": "assistant",
                            "content": friendly_assistant_error(exc),
                        }
                    )
                st.rerun(scope="fragment")

            if messages and st.button(
                "Clear conversation",
                icon=":material/refresh:",
                key="assistant_clear",
                use_container_width=True,
            ):
                st.session_state.data_assistant_messages = []
                st.rerun(scope="fragment")


# 15. Sidebar filters

def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str, int, str, list[str], str]:
    """Apply sidebar filters."""
    st.sidebar.header("Filters")

    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Dashboard page",
        [
            "Overview",
            "Route Explorer",
            "Ship and Vessel Analysis",
            "Port Analysis",
            "Emission Intensity",
            "Animated Route Map",
        ],
    )

    metric_options = ["Total CO2", "CO2 per km", "CO2 per DWT-km", "Voyage count"]
    metric_name = "Total CO2"
    top_n = 20
    route_mode = "Country pair"
    active_filters: list[str] = []
    filtered = df.copy()

    if page in {
        "Route Explorer",
        "Ship and Vessel Analysis",
        "Port Analysis",
        "Emission Intensity",
        "Animated Route Map",
    }:
        st.sidebar.subheader("View controls")

    if page == "Route Explorer":
        metric_name = st.sidebar.selectbox(
            "Ranking metric",
            metric_options,
            format_func=format_metric_label,
            help="Routes in this page are ordered by this metric.",
        )
        route_mode = st.sidebar.radio(
            "Analysis level",
            ["Country pair", "Directional country route", "Port-to-port route"],
            help="Choose the route granularity for the Route Explorer chart.",
        )
        top_n = st.sidebar.slider("Routes shown", min_value=5, max_value=80, value=10, step=5)

    elif page == "Ship and Vessel Analysis":
        metric_name = st.sidebar.selectbox(
            "Comparison metric",
            metric_options,
            format_func=format_metric_label,
            help="Ship types are compared using this metric.",
        )
        top_n = st.sidebar.slider("Ship types shown", min_value=5, max_value=80, value=20, step=5)

    elif page == "Port Analysis":
        top_n = st.sidebar.slider(
            "Ports shown",
            min_value=5,
            max_value=80,
            value=20,
            step=5,
            help="Ports are ranked by total CO₂.",
        )

    elif page == "Emission Intensity":
        metric_name = st.sidebar.selectbox(
            "Intensity measure",
            ["CO2 per km", "CO2 per DWT-km"],
            format_func=format_metric_label,
            help=(
                "Choose distance-normalised intensity (kg/km) or capacity-adjusted "
                "intensity (kg/DWT-km). The latter includes only records with positive DWT."
            ),
        )

    elif page == "Animated Route Map":
        metric_name = st.sidebar.selectbox(
            "Map metric",
            metric_options,
            format_func=format_metric_label,
            help="Map line colour, width, and route ranking use this metric.",
        )
        top_n = st.sidebar.slider("Routes shown", min_value=5, max_value=80, value=20, step=5)

    st.sidebar.subheader("Time and scope")
    years = sorted([int(x) for x in df["departure_year"].dropna().unique()])
    selected_years = st.sidebar.multiselect("Departure year", years, default=years)
    if selected_years:
        filtered = filtered[filtered["departure_year"].isin(selected_years)]
        if len(selected_years) != len(years):
            active_filters.append(f"Years: {', '.join(map(str, selected_years))}")

    scopes = sorted(df["voyage_scope"].dropna().astype(str).unique())
    selected_scopes = st.sidebar.multiselect("Voyage scope", scopes, default=scopes)
    if selected_scopes:
        filtered = filtered[filtered["voyage_scope"].isin(selected_scopes)]
        if len(selected_scopes) != len(scopes):
            active_filters.append(f"Voyage scope: {', '.join(selected_scopes)}")

    st.sidebar.subheader("Route selection")
    with st.sidebar:
        selected_country_pair = searchable_route_select(
            "Country pair route",
            ranked_filter_options(filtered, "country_pair"),
            key="country_pair_searchbox",
            placeholder="Search or select All...",
            display_func=format_country_pair_option,
            default_limit=60,
            search_limit=250,
            help_text=(
                "Searches all current country-pair routes. Initially lists the top "
                "60 by voyage count and returns up to 250 matches."
            ),
        )
    if selected_country_pair != "All":
        filtered = filtered[filtered["country_pair"].astype(str).eq(selected_country_pair)]
        active_filters.append(f"Country pair: {format_country_pair_option(selected_country_pair)}")

    if page not in {"Route Explorer", "Port Analysis", "Emission Intensity"}:
        with st.sidebar:
            selected_port_route = searchable_route_select(
                "Port-to-port route",
                ranked_filter_options(filtered, "port_route_display"),
                key="port_route_searchbox",
                placeholder="Search or select All...",
                default_limit=80,
                search_limit=500,
                help_text=(
                    "Searches all current port-to-port routes. Initially lists the top "
                    "80 by voyage count and returns up to 500 matches."
                ),
            )
        if selected_port_route != "All":
            filtered = filtered[filtered["port_route_display"].astype(str).eq(selected_port_route)]
            active_filters.append(f"Port route: {selected_port_route}")

    st.sidebar.subheader("Ship selection")
    ship_type_options = ["All"] + ranked_filter_options(filtered, "ship_type_clean")
    selected_ship_type = st.sidebar.selectbox(
        "Ship type focus",
        ship_type_options,
        help=(
            "Uses the voyage inventory's analysis-ready ship classification. "
            "Use this as the main filter for emissions and ship-type comparisons."
        ),
    )
    if selected_ship_type != "All":
        filtered = filtered[filtered["ship_type_clean"].astype(str).eq(selected_ship_type)]
        active_filters.append(f"Ship type: {selected_ship_type}")

    if "vessel_general_type" in df.columns:
        vessel_types = ["All"] + ranked_filter_options(filtered, "vessel_general_type")
        selected_vessel_type = st.sidebar.selectbox(
            "Kpler vessel type",
            vessel_types,
            format_func=format_kpler_vessel_type,
            help=(
                "Uses the broader Kpler vessel category joined by IMO. It supplements "
                "the voyage-based ship classification. When both ship filters are "
                "selected, both conditions are applied."
            ),
        )
    else:
        selected_vessel_type = "All"
    if selected_vessel_type != "All":
        filtered = filtered[filtered["vessel_general_type"].astype(str).eq(selected_vessel_type)]
        active_filters.append(f"Kpler vessel type: {selected_vessel_type}")

    return filtered, metric_name, top_n, page, active_filters, route_mode


# 16. Main app controller

def render_dashboard_partner_footer() -> None:
    """Render compact institutional marks and the data provenance note."""
    st.markdown(
        f"""
        <footer class="dashboard-partner-footer" aria-label="Project partners and data source">
            <div class="dashboard-footer-logos" aria-label="Project institutions">
                <div class="dashboard-footer-logo logo-ntnu">
                    <img src="{NTNU_LOGO_DATA_URI}" alt="Norwegian University of Science and Technology">
                </div>
                <div class="dashboard-footer-logo logo-tyndall">
                    <img src="{TYNDALL_LOGO_DATA_URI}" alt="Tyndall Centre for Climate Change Research">
                </div>
                <div class="dashboard-footer-logo logo-ambs">
                    <img src="{AMBS_LOGO_DATA_URI}" alt="Alliance Manchester Business School">
                </div>
            </div>
            <p class="dashboard-footer-source"><strong>Data source:</strong> The data presented in this dashboard was provided by the research team at the Norwegian University of Science and Technology (NTNU) and generated using their MariTEAM model.</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    """Run the Streamlit dashboard."""
    st.title("North Sea Shipping Emissions Explorer")

    with st.sidebar:
        st.header("Data Source")
        parquet_path = st.text_input("Enriched parquet file path", value=DEFAULT_PARQUET_PATH)

    try:
        df = load_data(parquet_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    render_data_assistant(df)

    filtered_df, metric_name, top_n, page, active_filters, route_mode = apply_filters(df)

    if filtered_df.empty:
        st.warning("No records match the current filters.")
        st.stop()

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} voyages. "
        f"Departure range: {filtered_df['departure_time'].min()} "
        f"to {filtered_df['departure_time'].max()}."
    )
    render_selection_summary(active_filters, page, metric_name, top_n, route_mode)

    if page == "Overview":
        show_overview(filtered_df, df)
    elif page == "Route Explorer":
        show_route_explorer(filtered_df, metric_name, top_n, route_mode)
    elif page == "Ship and Vessel Analysis":
        show_ship_analysis(filtered_df, metric_name, top_n)
    elif page == "Port Analysis":
        show_port_analysis(filtered_df, top_n)
    elif page == "Emission Intensity":
        show_intensity_analysis(filtered_df, metric_name)
    elif page == "Animated Route Map":
        show_animated_route_map(filtered_df, top_n, metric_name)

    render_dashboard_partner_footer()

# 17. Run the Streamlit app

if __name__ == "__main__":
    main()
