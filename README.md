### Interim Report for Telecom Data Analysis Project

##### Project Objectives

This project aims to provide insights into user behaviors and engagement metrics using a telecom dataset. The analysis is divided into two major tasks:

  + **Task 1: User Overview Analysis**
  + **Task 2: User Engagement Analysis**

The outcomes of these tasks will guide strategic decisions to enhance customer satisfaction and optimize network resources.

##### Task 1: User Overview Analysis

**Exploratory Data Analysis (EDA)**

**Subtasks:**

1. **Top 10 Handsets Used by Customers:**

 + Identified the most frequently used handsets in the dataset.

2. **Top 3 Handset Manufacturers:**

 + Apple, Samsung, and Huawei were found to be the leading manufacturers.

3. **Top 5 Handsets per Top 3 Manufacturers:**

 + **Apple**:

    1. iPhone 6S (A1688) - 9,391 users

    2. iPhone 6 (A1586) - 8,991 users

    3. iPhone 7 (A1778) - 6,274 users

    4. iPhone SE (A1723) - 5,165 users

    5. iPhone 8 (A1905) - 4,977 users

 + **Samsung**:

    1. Galaxy S8 (SM-G950F) - 4,459 users

    2. Galaxy A5 (SM-A520F) - 3,699 users

    3. Galaxy J5 (SM-J530) - 3,674 users

    4. Galaxy J3 (SM-J330) - 3,454 users

    5. Galaxy S7 (SM-G930X) - 3,169 users

 + **Huawei:**

    1. B528S-23A - 19,724 users

    2. E5180 - 2,073 users

    3. P20 Lite/Nova 3E - 2,011 users

    4. P20 - 1,475 users

    5. Y6 2018 - 987 users

**User Behavior Analysis on Applications**

 + **Aggregated metrics per user**:

    + Number of sessions

    + Total session duration

    + Total download (DL) and upload (UL) traffic

    + Total data volume

**Statistical and Graphical Analyses**:

1. **Descriptive Analysis**:

 + Variables were segmented into deciles based on session duration. The total data (DL + UL) was calculated for each decile.

2. **Univariate Analysis**:

 + Statistical metrics such as mean, median, and standard deviation were computed.

 + Visualizations included histograms and box plots to detect outliers and distributions.

3. **Bivariate and Correlation Analysis:**

 + Relationships between applications (e.g., Social Media, Google, YouTube) and total data usage were explored.

 + A correlation matrix revealed significant associations among variables.

4. **Dimensionality Reduction**:

 + Principal Component Analysis (PCA) identified key dimensions influencing user behavior, reducing redundant variables.

**Recommendations to Marketing Teams**

1. **Targeted Campaigns**: Leverage insights about popular handsets for tailored marketing.

2. **Collaborations**: Partner with top manufacturers for co-branding opportunities.

3. **Feedback Mechanisms**: Collect ongoing feedback on popular devices and applications.

##### Task 2: User Engagement Analysis

**Engagement Metrics**:

1. ****Session Frequency****

2. ****Session Duration****

3. ****Total Traffic (DL + UL)****

**Key Analyses:**

 1. **Top 10 Customers Per Metric**:

    + Ranked customers based on session frequency, duration, and total traffic.

 2. **K-Means Clustering**:

    + Optimized number of clusters (‘k’) identified using the elbow method.

    + Normalized metrics were used for clustering.

    + Customers were grouped into three engagement levels:

        + Low Engagement

        + Moderate Engagement

        + High Engagement

 3. **Cluster Statistics**:

    + Minimum, maximum, average, and total metrics per cluster were calculated.

 4. **Application-Specific Engagement**:

    + Aggregated user traffic per application.

    + Identified top 10 users for each application.

 5. **Visualization**:

    + Bar charts and scatter plots depicted engagement distribution and application usage trends.

**Insights and Recommendations**:

 1. **Network Optimization**:

    + Prioritize resource allocation for high-traffic applications like Social Media and YouTube.

 2. **Customer Retention Strategies**:

    + Engage top users with loyalty programs or data rewards.

 3. **Improved Service Delivery**:

    + Analyze engagement patterns to predict future demands and tailor services accordingly.

**Data Cleaning and Handling Missing Values**

 1. **Key Identifiers**:

    + Rows with missing data in critical columns (e.g., Bearer Id, IMSI, MSISDN) were removed.

 2. **Performance Metrics (RTT, DL TP, UL TP)**:

    + Missing values were replaced with the mean or median.

**Impact of Data Cleaning**:

+ Ensured dataset integrity while retaining valuable insights.

**Next Steps**

1. **Deep-Dive Analysis**:

    + Expand engagement metrics to include latency and application-specific behavior.

2. **Predictive Modeling**:

    + Develop models to forecast user behavior and engagement trends.

3. **Dynamic Recommendations**:

    + Implement real-time suggestions for network optimization and customer targeting.