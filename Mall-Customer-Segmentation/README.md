# Customer Segmentation

A customer segmentation project using **K-Means clustering** to identify distinct customer groups based on **annual income** and **spending score**.

The project applies unsupervised learning techniques to discover patterns in customer behavior and translate the resulting clusters into meaningful customer segments.

## Project Overview

Customer segmentation is a useful technique for understanding different types of customers based on their characteristics and behavior.

In this project, the **Mall Customer dataset** is analyzed to answer questions such as:

* Which customers have high spending scores?
* Are high-income customers always high spenders?
* What distinct customer groups can be identified?
* How can these groups be used for targeted marketing?

The main clustering algorithm used is **K-Means**, with **DBSCAN** explored as an additional approach.

## Objectives

* Explore and clean the customer dataset
* Analyze the relationship between income and spending score
* Scale the clustering features
* Determine an appropriate number of clusters using the **Elbow Method**
* Apply **K-Means clustering**
* Visualize customer segments in 2D
* Analyze the characteristics of each cluster
* Compare average spending between customer groups
* Explore **DBSCAN** as an alternative clustering method
* Extract practical business insights from the resulting segments

## Dataset

**Mall Customer Dataset**

The dataset contains customer information including:

* `CustomerID`
* `Gender`
* `Age`
* `Annual Income (k$)`
* `Spending Score (1-100)`

For the main clustering analysis, the following features are used:

* **Annual Income**
* **Spending Score**

These variables provide a simple way to analyze customers according to their purchasing behavior and income level.

## Methodology

### 1. Data Exploration

The dataset is first inspected to understand:

* Dataset structure and dimensions
* Data types
* Missing values
* Duplicate records
* Descriptive statistics

### 2. Exploratory Data Analysis

The relationship between **Annual Income** and **Spending Score** is explored through visualizations, including distributions and scatter plots.

The scatter plot provides an initial view of possible natural customer groupings.

### 3. Feature Scaling

The selected features are standardized using `StandardScaler`.

Scaling is important because K-Means relies on distance calculations. Without scaling, a feature with a larger numerical range could have a disproportionate effect on the clustering process.

### 4. Finding the Optimal Number of Clusters

The **Elbow Method** is used to evaluate different values of `K`.

Inertia is calculated for several cluster counts, and the resulting values are visualized to identify a suitable point where adding additional clusters provides diminishing improvement.

### 5. K-Means Clustering

After selecting an appropriate number of clusters, the final K-Means model is trained.

Each customer receives a cluster label, allowing the dataset to be analyzed according to customer segments.

### 6. Cluster Visualization

The resulting clusters are visualized using a 2D scatter plot:

* **X-axis:** Annual Income
* **Y-axis:** Spending Score
* **Color:** Customer Cluster

Cluster centroids are also displayed to make the separation between customer groups easier to interpret.

## Customer Segment Analysis

After clustering, each segment is analyzed using:

* Number of customers
* Average annual income
* Average spending score

Based on these characteristics, clusters can be described using behavioral categories such as:

| Customer Type            | Income | Spending |
| ------------------------ | ------ | -------- |
| High-Value Customers     | High   | High     |
| Potential Customers      | High   | Low      |
| Frequent Spenders        | Low    | High     |
| Low-Engagement Customers | Low    | Low      |
| Moderate Customers       | Medium | Medium   |

The final segment names are determined from the actual clustering results rather than being predefined.

## Average Spending Analysis

Average spending scores are calculated for each cluster and visualized using a bar chart.

This helps identify which customer groups demonstrate stronger spending behavior and provides another perspective for comparing the segments.

## Bonus: DBSCAN

As an additional experiment, **DBSCAN** is applied to the scaled customer data.

DBSCAN is useful for exploring clustering based on density rather than requiring the number of clusters to be specified beforehand.

The results are compared with K-Means to observe:

* Differences in cluster formation
* Noise or outlier points
* Sensitivity to clustering parameters
* How different unsupervised learning methods interpret the same customer data

## Business Insights

Customer segmentation can help a retail business move beyond treating all customers in the same way.

For example, different segments can be targeted with different strategies:

* **High-income, high-spending customers** → loyalty programs and premium offers
* **High-income, low-spending customers** → personalized promotions to increase engagement
* **Low-income, high-spending customers** → discounts and loyalty incentives
* **Low-income, low-spending customers** → low-cost promotional campaigns
* **Moderate customers** → strategies designed to gradually increase spending

The purpose is not only to identify clusters, but also to understand what those clusters mean from a business perspective.

## Technologies & Libraries

* **Python**
* **Pandas** — Data manipulation and analysis
* **Matplotlib** — Data visualization
* **Scikit-learn** — Feature scaling and clustering
* **K-Means** — Main clustering algorithm
* **DBSCAN** — Bonus clustering algorithm

## Project Structure

```text
Customer-Segmentation/
│
├── customer_segmentation.ipynb
├── Mall_Customers.csv
└── README.md
```

## Key Concepts

* Unsupervised Learning
* Customer Segmentation
* K-Means Clustering
* DBSCAN
* Feature Scaling
* Elbow Method
* Cluster Analysis
* Data Visualization
* Business Analytics

## Conclusion

This project demonstrates how unsupervised machine learning can be used to discover meaningful customer segments without predefined labels.

By combining exploratory data analysis, feature scaling, K-Means clustering, visualization, and cluster-level analysis, customer behavior can be transformed into actionable business insights.
