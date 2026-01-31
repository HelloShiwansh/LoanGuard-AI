# Dataset Description – CreditWise Loan Approval System

## Overview
The dataset contains historical loan application records from a mid-sized financial institution operating across urban and rural regions of India.  
Each row represents **one loan applicant** and captures personal, financial, employment, and credit-related attributes used to determine loan approval.

The dataset is intended for training a **binary classification Machine Learning model** to predict whether a loan should be **Approved (1)** or **Rejected (0)**.

---

## Target Variable
- **Loan_Approved**
  - `1` → Loan Approved  
  - `0` → Loan Rejected  

---

## Feature Description

| Column Name | Description |
|------------|-------------|
| Applicant_ID | Unique identifier for each loan applicant |
| Applicant_Income | Monthly income of the applicant |
| Coapplicant_Income | Monthly income of the co-applicant (if any) |
| Employment_Status | Employment type (Salaried / Self-Employed / Business) |
| Age | Age of the applicant in years |
| Marital_Status | Marital status (Married / Single) |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score of the applicant |
| Existing_Loans | Number of currently active loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Current savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration in months |
| Loan_Purpose | Purpose of loan (Home / Education / Personal / Business) |
| Property_Area | Property location (Urban / Semi-Urban / Rural) |
| Education_Level | Education level (Undergraduate / Graduate / Postgraduate) |
| Gender | Gender of the applicant (Male / Female) |
| Employer_Category | Employer type (Government / Private / Self) |
| Loan_Approved | Target variable indicating approval status |

---

## Dataset Characteristics
- **Problem Type:** Supervised Learning  
- **Task:** Binary Classification  
- **Data Type:** Structured tabular data  
- **Potential Issues:**  
  - Missing values  
  - Class imbalance  
  - Categorical feature encoding  
  - Outliers in income and loan amounts  

---

## Intended Use
This dataset will be used to:
- Train and evaluate Machine Learning models
- Learn hidden patterns from historical data
- Reduce bias and inconsistency in manual loan approval processes
- Provide fast and accurate loan approval recommendations before human verification
