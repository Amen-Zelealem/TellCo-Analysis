# 10 Academy: Artificial Intelligence Mastery


# Telecom Analytics Project

## Project Overview

This project is focused on performing a comprehensive analysis of user behavior, engagement, experience, and satisfaction in a telecom dataset. The project is designed with modular, reusable code and features a Streamlit dashboard for data visualization. The key objectives include:

- **User Overview Analysis**: Analyze handset usage, handset manufacturers, and application usage.
- **User Engagement Analysis**: Track user engagement across different applications and cluster users based on engagement metrics.
- **Experience Analytics**: Assess user experience based on network parameters and device characteristics.
- **Satisfaction Analysis**: Calculate and predict user satisfaction scores based on engagement and experience.

The project structure is organized to support reproducible and scalable data processing, modeling, and visualization.

## Project Structure

```plaintext
├── .github/
│   └── workflows/
│       ├── unittests.yml            # GitHub Actions workflow for running unit tests
├── .vscode/
│   └── settings.json                # Configuration for VSCode environment
├── app/
│   ├── main.py
│   ├── README.md
├── notebooks/
│   ├── __init__.py
│   ├── user_engagment_analysis.ipynb    # Jupyter notebook for engagment_analysis
│   ├── user_experince_analysis.ipynb          # Jupyter notebook for user_experince_analysis
│   ├── user_overview_analysis.ipynb          # Jupyter notebook for user_overview_analysis
│   ├── user_satisfaction_analysis.ipynb          # Jupyter notebook for user_satisfaction_analysis
│   ├── README.md                     # Description of notebooks
├── scripts/
│   ├── __init__.py
│   ├── db_conn.py           # Script for database connection
│   ├── eda_pipeline.py              # EDA steps implemented 
│   ├── experience_analytics.py      # User experience analytics module
│   ├── handset_analysis.py          # Handset analysis module
│   ├── handset_dashboard.py         # Streamlit dashboard for visualizing top handsets
│   ├── user_satisfaction_analytics.py   # Machine learning models and training scripts for predicting satisfaction score
│   ├── user_satisfaction_dashabord.py     # Streamlit dashboard script
│   ├── user_analysis.py               # User analysis functions
│   ├── user_engagement_analysis.py     # User engagement analytics module
│   ├── user_engagement_dashboard.py     # User engagement dashboard module
│   ├── README.md
└── src/
    ├── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_user_engagement_analysis.py      # Unit tests for user engagement module
│   ├── test_user_analysis.py         # Unit tests for user analysis module
│   ├── test_handset_analysis.py      # Unit tests for handset analysis module
│   ├── test_experience_analysis.py  # Unit tests for user experience module
│   ├── test_eda_pipeline.py          # Unit tests for EDA pipeline module  
├── .gitignore                        # Files and directories to be ignored by Git
├── Dockerfile                        # Instructions to build a Docker image
├── README.md                         # Project overview and instructions
├── requirements.txt                  # List of dependencies for the project