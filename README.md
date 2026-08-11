# Skin Lesion Educational Assistant  Azure AI Foundry Agentic System

A multi-modal, tool-using AI agent built on Azure AI Foundry that gives educational, plain-language context about skin lesion photos, combining a custom-trained computer vision model with an LLM-based reasoning layer. Built as a portfolio project to explore agentic AI orchestration on native Azure tooling, as a companion piece to Azure ML-focused work.

**This is an educational demo, not a medical device.** It never provides a diagnosis and always directs the user to consult a dermatologist. See [Responsible use](#responsible-use) below.

## Architecture

```
User uploads image + context (age, location, duration, changes)
        │
        ▼
Streamlit frontend
        │
        ▼
Azure AI Foundry Agent (GPT-5-mini)
        │
        ├──► Tool: analyze_skin_lesion_image  ──► Azure Custom Vision endpoint
        │                                          (trained on HAM10000 dataset)
        │
        └──► Tool: explain_abcde_rule  ──► static educational reference
        │
        ▼
Plain-language, cautious response, always recommending professional consultation
```

## What this project covers

- **Data**: HAM10000 dermatoscopic image dataset (10,015 images, 7 diagnostic categories), stored and versioned in Azure Blob Storage / Azure ML Data Assets, used strictly for non-commercial, educational purposes per its license
- **Computer vision**: an Azure Custom Vision multiclass classifier trained on a balanced sample of the dataset (Precision 86%, Recall 74.3%, AP 87.2%)
- **Agent orchestration**: an Azure AI Foundry agent (GPT-5-mini) that decides when to call the vision tool, interprets the raw prediction, and produces a careful, human-readable response  this is the core of the project, not the vision model itself
- **Tool use**: the agent has two tools  one that classifies the uploaded image, and one that explains the ABCDE self-examination guideline on request
- **Safety-first prompt design**: the agent is explicitly instructed to never state a diagnosis, to disclose that it's an automated tool trained on limited data, and to always recommend professional consultation, with extra emphasis when the top prediction is a higher-risk category (e.g. melanoma)
- **Frontend**: a Streamlit app for image upload and free-text context input, calling the agent live

## Example interaction

> Uploaded image → top prediction: melanoma (43.0%), melanocytic nevus (34.7%), benign keratosis-like lesion (13.4%)
>
> *"The tool's top prediction is melanoma (43.0% confidence), followed by melanocytic nevus (34.7%) and benign keratosis-like lesion (13.4%), but this is not a medical diagnosis. This is an automated pattern-recognition model trained on a limited dataset and cannot replace an in-person clinical evaluation. [...] Because the top prediction is melanoma, please consult a dermatologist or doctor for an accurate assessment."*

This example was deliberately chosen because the model's predictions were close together  the agent surfaced that uncertainty rather than presenting the top guess with false confidence, and escalated its caution because of the higher-risk category.

## Tech stack

Python, Azure AI Foundry (Agent Service, GPT-5-mini), Azure Custom Vision, Azure Blob Storage, Azure Machine Learning (Data Assets), Streamlit, Docker

## Repository structure

```
app.py              Streamlit frontend + agent orchestration logic
requirements.txt     Python dependencies
Dockerfile           Container definition for the app
notebooks/            Data collection, Custom Vision training, and agent setup notebooks (Azure ML Studio)
```

## Running locally

```bash
uv venv
uv add streamlit azure-ai-agents azure-identity azure-cognitiveservices-vision-customvision python-dotenv pillow
az login
uv run streamlit run app.py
```

Requires a `.env` file with `PROJECT_ENDPOINT`, `AGENT_ID`, `CV_PREDICTION_ENDPOINT`, `CV_PREDICTION_KEY`, `CV_PROJECT_ID`, `CV_PUBLISHED_NAME`.

## Responsible use

This project is built for educational and technical demonstration purposes only:
- It is **not a diagnostic tool** and must never be used to make real health decisions
- The underlying vision model was trained on a small, class-balanced sample (100 images/category) using Quick Training on Azure Custom Vision's free tier  it is a proof of concept, not a validated clinical model
- The HAM10000 dataset is used under its non-commercial license, for research/educational purposes only
- The agent's instructions explicitly and consistently redirect the user toward professional medical consultation

## Why I built this

Built to explore Azure's agentic AI stack (Azure AI Foundry, tool-calling, multi-modal input) in a domain that required careful, responsible prompt design  not just wiring a model to an API, but thinking through what an AI system should and shouldn't say when the stakes are personal. This complements a separate Azure ML project (Premier League match prediction) that focuses on the classic ML/MLOps side of Azure.
