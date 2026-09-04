# North Sea Shipping Emissions Explorer

An interactive Streamlit dashboard for exploring shipping activity and
emissions in the North Sea region. The dashboard supports analysis across
voyages, routes, ports, vessel types, total CO2 emissions, and emission
intensity.

## Repository contents

```text
NorthSea_Emissions_Dashboard_GitHub/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── data/
│   └── voyages_enriched.parquet    # supplied separately
├── .gitignore
├── README.md
├── USER_MANUAL.md
├── requirements.txt
└── streamlit_app.py
```

Detailed macOS and Windows installation, dataset-placement, launch and
troubleshooting instructions are provided in [`USER_MANUAL.md`](USER_MANUAL.md).

This repository contains the dashboard code and configuration only. Raw source
files, data-integration notebooks, diagnostic outputs, and Unity development
files are not required to run this version of the dashboard.

## Data access

The required dataset is not included in this GitHub repository because of its
file size and data-access considerations. It will be shared separately with
authorised project collaborators through WeTransfer.

After downloading the WeTransfer file, place it in the following location
without changing its filename:

```text
NorthSea_Emissions_Dashboard_GitHub/data/voyages_enriched.parquet
```

The final project structure must therefore include:

```text
data/
└── voyages_enriched.parquet
```

The dashboard uses a relative path, so no user-specific file path needs to be
edited.

## Local setup

Python 3.10 or later is recommended.

### macOS or Linux

Open Terminal in the project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

### Windows

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Optional local AI assistant

The main dashboard works without an AI service. The optional Maritime Data
Assistant uses a local Ollama model to interpret questions and DuckDB to query
the full Parquet dataset locally. The dataset is not uploaded to an external AI
provider.

To enable the assistant, install Ollama and run:

```bash
ollama pull qwen3:4b-instruct
ollama serve
```

The default local model settings are provided in:

```text
.streamlit/secrets.toml.example
```

If custom settings are required, copy this file to
`.streamlit/secrets.toml`. Do not commit the completed secrets file.

## Updating the dashboard

Dashboard changes can be made in `streamlit_app.py`, tested locally, and pushed
to this GitHub repository. Collaborators can receive the latest version by
pulling or downloading the updated repository. The separately supplied Parquet
file does not need to be downloaded again unless the dataset itself changes.
