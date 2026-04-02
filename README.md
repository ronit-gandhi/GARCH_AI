Overview

GARCH_AI is a real-time financial analysis application that combines Reddit sentiment, volatility modeling, and a local AI assistant to provide insights into stock behavior.

Objectives
	•	Analyze social sentiment for financial assets
	•	Model market risk using GARCH
	•	Provide AI-driven explanations of results

Data Sources
	•	Reddit (PRAW API)
	•	Yahoo Finance

Methodology
	•	Sentiment analysis using FinBERT
	•	GARCH(1,1) volatility modeling
	•	Integration with a local language model for interpretation

Repository Structure
	•	app.py: Streamlit interface
	•	sentiment.py: NLP pipeline
	•	garch.py: Volatility modeling
	•	ai_module.py: LLM-based explanations

Key Features
	•	Real-time sentiment analysis
	•	Volatility estimation and forecasting
	•	AI-generated explanations of market behavior
	•	Interactive user interface

How to Run
	1.	Install dependencies
	2.	Set API credentials
	3.	Launch Streamlit app

Limitations
	•	Real-time data constraints
	•	Model assumptions in GARCH
	•	Dependence on external APIs

Future Work
	•	Improve model interpretability
	•	Add multi-asset support
	•	Incorporate causal inference layer
	•	Deploy scalable backend

Purpose

This project demonstrates integration of financial modeling, NLP, and AI-based explanation systems.
