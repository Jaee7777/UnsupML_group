**Exploratory Data Analysis (EDA) For Online Retail Dataset**

**Dataset Overview**

The Online Retail dataset is a transactional record from an online retail store, containing detailed information on customer purchases over time. It includes the following key attributes:
- InvoiceNo: A unique identifier for each transaction or order made.
- StockCode: The specific code assigned to each product sold.
- Description: A brief detail or name of the product purchased.
- Quantity: The number of units sold for each product in the transaction.
- InvoiceDate: The exact date and time when the transaction occurred.
- UnitPrice: The cost of one unit of the product.
- CustomerID: A unique identifier assigned to each customer.
- Country: The country where the customer is located.

This dataset enables in-depth analysis of sales trends, customer purchasing behavior, product performance, and geographic distribution. It is suitable for customer segmentation through RFM (Recency, Frequency, Monetary) analysis, identifying high-value customers, tracking sales performance, and uncovering seasonal trends or purchasing patterns.


**Initial Data Inspection**

Once the dataset is loaded, an initial preview is performed to understand its structure. This includes checking the first few rows of the dataset to get a sense of the columns and values it contains. The dataset consists of 541,909 rows and 8 columns. Additionally, an inspection reveals missing values, remarkably in the CustomerID column, with approximately 24.9% of entries missing. The Description field also contains some null values. This step helps in identifying any immediate irregularities, such as unexpected column names, missing headers, or structural inconsistencies, and sets the stage for further data cleaning and analysis.

**Data Pre-Processing**

**Handling Missing Data**
Columns that contain a significant number of missing values but are essential for analysis are carefully addressed. In this dataset, missing values in key columns such as CustomerID and Description are handled to ensure data quality. Specifically, rows with missing CustomerID are removed, as they are crucial for customer-centric analyses like RFM segmentation. Similarly, rows with missing Description are dropped to maintain the integrity of product-level insights. This cleaning process reduces the dataset size from 541,909 to approximately 397,884 rows, ensuring that the remaining data is complete and reliable for further analysis.

**Filtering Out Invalid Transactions**
Retail transactions sometimes contain negative or zero values for product quantities and unit prices, often indicating returns or incorrect data entry. These records are removed from the dataset to ensure that analyses reflect actual purchases rather than returns or erroneous entries. By removing these anomalies, the dataset becomes more reliable for further exploration.

**Computing RFM (Recency, Frequency, Monetary) Metrics**
One of the fundamental aspects of customer behavior analysis in e-commerce is the Recency-Frequency-Monetary (RFM) framework:
- Recency measures the number of days since a customer’s last purchase. Customers who made recent purchases are generally more engaged.
- Frequency counts the number of unique transactions made by each customer. High-frequency customers tend to be more loyal and valuable.
- Monetary calculates the total amount spent by each customer over the recorded period. Customers with high monetary value contribute significantly to overall revenue. These metrics are computed by grouping transactions by customer ID and aggregating purchase behaviors accordingly.

![OnlineRetail-Heatmap](../fig/OnlineRetail_heatmap.png)


**Customer Segmentation Score**
The customer segmentation scoring process involves evaluating each customer based on three key metrics: Recency (how recently a customer made a purchase), Frequency (how often they purchase), and Monetary (the total amount they spend). 
Each metric is scored on a scale, typically from 1 to 4 or 1 to 5, using quantiles—where lower Recency scores indicate recent engagement, and higher Frequency and Monetary scores reflect loyal and high-spending customers. These individual scores are then combined to create an overall RFM score, which helps categorize customers into meaningful segments. 
For example, customers with high scores across all three metrics are labeled as Loyal Customers, those with low recency but high frequency and monetary values may be At-Risk, while customers with low scores across all metrics may be considered Churned. This initial segmentation enables us to find an overview of customer preferences and provide the targeted marketing strategies to retain loyal customers, re-engage at-risk ones, and convert new buyers into frequent purchasers.

![OnlineRetail-SegmentationScore](https://github.com/user-attachments/assets/31505b87-9e41-4753-a1f3-c33863545fe0)



The above graph shows that the business has a healthy loyal customer base but should focus on reactivating "At Risk" customers and nurturing "Potential Loyalists" to expand the loyal segment.

**Log Transformation for Data Normalization**
Since this dataset is extremely skewed, log transformation is applied. This helps stabilize the data by reducing the effect of outliers. Without transformation, customers with extremely high spending might skew statistical models.
The logarithmic transformation ensures that all three RFM metrics (Recency, Frequency, and Monetary) have a more balanced distribution, making them more suitable for clustering algorithms.

![OnlineRetail-3Ddistribution](https://github.com/user-attachments/assets/bcecb48c-56e3-40a3-ba54-90d4e3aa81b8)



**Scaling the Data using Min-Max Normalization**
After log transformation, the data is further processed using Min-Max Scaling. This step ensures that all three variables—Recency, Frequency, and Monetary—are scaled to a common range between 0 and 1. This normalization is crucial for distance-based clustering algorithms like Gaussian Mixture Models (GMM), as it prevents features with larger ranges from dominating the clustering process.
For DBSCAN (Density-Based Spatial Clustering of Applications with Noise), Standard Scaler is preferred over Min-Max scaling because DBSCAN relies on the concept of density and uses Euclidean distance to determine the proximity of points. Standard Scaler transforms the data to have a mean of 0 and a standard deviation of 1, which helps in maintaining the natural distribution of the data without compressing outliers, as Min-Max scaling would. 

Since DBSCAN is sensitive to the density of points, using Standard Scaler ensures that clusters are identified based on natural data density rather than being influenced by the compressed scales of Min-Max normalization. This approach allows DBSCAN to better detect dense regions and correctly identify noise or outliers.


**Principal Component Analysis (PCA)**
We applied Principal Component Analysis (PCA) to reduce the dimensionality of a standardized RFM dataset to two principal components. Using scikit-learn, it extracts the two components that capture the most variance in the data, transforming the original features into a new coordinate system. The result is then converted into a DataFrame with columns labeled "Recency" and "Frequency", representing the projections onto the first two principal components. This transformation simplifies the data for visualization or clustering while retaining as much variance as possible from the original dataset.

![OnlineRetail-PCAdistribution](https://github.com/user-attachments/assets/1863f972-a08f-4c03-86a3-b3b0f66af93e)


**Goal Of the Project**
The goal of this project is to perform customer segmentation using Recency and Frequency data to better understand customer engagement patterns and identify distinct customer groups. By analyzing how recently and how often customers interact or make purchases, the project aims to classify customers into segments such as Loyal Customers, Partial Customers and At-Risk Customers. This segmentation is helpful for corporates to design targeted marketing strategies, improve customer retention, and optimize revenue generation. 

