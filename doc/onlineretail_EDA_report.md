# EDA for Online Retail

## Introduction
This document outlines the exploratory data analysis (EDA) performed on the Online Retail dataset. The goal of the analysis is to understand the dataset's structure, detect missing values, and identify any anomalies before proceeding to further modeling or insights.

## Data Overview
The dataset consists of the following key attributes:

- **InvoiceNo**: A unique identifier for each transaction.
- **StockCode**: A unique code assigned to each product.
- **Description**: A brief description of the product.
- **Quantity**: The number of items purchased in each transaction.
- **InvoiceDate**: The timestamp of the transaction.
- **UnitPrice**: The price per unit of the product.
- **CustomerID**: A unique identifier for each customer.
- **Country**: The country from which the transaction was made.

## Missing Values Analysis
One of the primary checks performed was identifying missing values in the dataset:

- **CustomerID** has a significant number of missing values.
- **Description** also contains some missing values.

To handle missing values:
1. CustomerID missing values can either be imputed if a pattern is found or removed if they do not contribute to analysis.
2. Description missing values may need cross-verification with `StockCode`.

## Summary Statistics
A statistical summary of numerical features was generated to check for:
- Distribution of `Quantity` and `UnitPrice`.
- Unusual values, such as negative `Quantity` or extremely high `UnitPrice`.

### Key Findings:
- There are transactions with negative `Quantity`, possibly indicating returns.
- Some items have a `UnitPrice` of zero, which may indicate promotional items or data entry errors.

## Data Distribution & Visualization
### Histograms
Histograms were used to inspect the distribution of:
- **Quantity**: Most transactions involve small purchases, with a few large orders.
- **UnitPrice**: Prices are heavily right-skewed, with a few very expensive items.

### Box Plots for Outliers
Box plots were used to detect outliers in:
- `Quantity`: Large positive values and negative values indicate potential data issues.
- `UnitPrice`: A few items have significantly high prices.

### Correlation Analysis
A correlation heatmap was generated to identify relationships between numerical features:
- `Quantity` and `UnitPrice` show little correlation, meaning price changes do not strongly affect order size.

## Handling Outliers
Outliers were identified in:
- `Quantity`: Negative values could be removed or analyzed separately as returns.
- `UnitPrice`: Extremely high values might be data entry errors.

## Data Cleaning & Transformation
To prepare the dataset for modeling:
1. **Remove or impute missing CustomerIDs.**
2. **Filter out negative `Quantity` values** unless specifically needed.
3. **Remove rows where `UnitPrice` is zero**, if they are errors.
4. **Apply MinMax scaling** for numerical columns before clustering or modeling.

## Conclusion
The EDA revealed:
- The presence of missing values, particularly in `CustomerID`.
- The need to address outliers in `Quantity` and `UnitPrice`.
- The dataset is heavily skewed, requiring transformations.
- **MinMax scaling is recommended** to normalize numerical features before clustering.

The next step is **feature engineering and clustering analysis** to segment customers and detect unusual buying patterns.
