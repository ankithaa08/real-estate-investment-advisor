# Comprehensive Exploratory Data Analysis (EDA) Report
**Project:** Real Estate Investment Advisor: Predicting Property Profitability & Future Value  
**Dataset:** India Housing Prices (250,000 records, 42 major cities, 20 states)  

---

## Section 1: Price & Size Analysis (Questions 1–5)

### Q1: What is the distribution of property prices?
- **Finding:** Property prices range from ₹10.0 Lakhs to ₹500.0 Lakhs with a **mean of ₹254.59 Lakhs** and a **median of ₹254.60 Lakhs**.
- **Distribution Shape:** Displays a well-balanced distribution across mid-income housing (₹30L–₹150L), upper-middle developments (₹150L–₹350L), and luxury residences (>₹350L).
- **Visualization:** `reports/figures/eda_01_price_distribution.png`

### Q2: What is the distribution of property sizes?
- **Finding:** Property sizes span from 500 sq.ft to 5,000 sq.ft with a **mean of 2,750 sq.ft** and **median of 2,749 sq.ft**.
- **Distribution Shape:** Gaussian-like distribution representing compact urban apartments up to sprawling independent houses and villas.
- **Visualization:** `reports/figures/eda_02_size_distribution.png`

### Q3: How does price per sq ft vary by property type?
- **Finding:** 
  - **Villas:** Command the highest median price per sq.ft (₹0.141 Lakhs/sq.ft = ₹14,100/sq.ft), reflecting premium land ownership and private amenities.
  - **Apartments:** Maintain a competitive median of ₹0.128 Lakhs/sq.ft (₹12,800/sq.ft), balancing density and common infrastructure.
  - **Independent Houses:** Average ₹0.123 Lakhs/sq.ft (₹12,300/sq.ft).
- **Visualization:** `reports/figures/eda_03_price_per_sqft_by_type.png`

### Q4: Is there a relationship between property size and price?
- **Finding:** Strong positive linear correlation (**Pearson r = 0.68**). As square footage increases, capital requirement scales proportionally, establishing size as the primary baseline determinant of absolute acquisition cost.
- **Visualization:** `reports/figures/eda_04_size_vs_price.png`

### Q5: Are there any outliers in price per sq ft or property size?
- **Finding:** Statistical IQR inspection detected positive outliers representing ultra-luxury penthouses and trophy assets in top-tier micro-markets. These were capped at the 99.5th percentile during preprocessing to protect gradient descent stability while preserving realistic asset spreads.
- **Visualization:** `reports/figures/eda_05_outlier_detection.png`

---

## Section 2: Location-Based Analysis (Questions 6–10)

### Q6: What is the average price per sq ft by state?
- **Finding:** High-commercialization states lead the price hierarchy:
  - **Top States:** Maharashtra, Karnataka, Tamil Nadu, and Delhi NCR command average rates exceeding ₹0.138 Lakhs/sq.ft (₹13,800/sq.ft).
  - **Emerging States:** Rajasthan, Madhya Pradesh, and Punjab offer higher affordability and value-entry headroom.
- **Visualization:** `reports/figures/eda_06_avg_price_per_sqft_by_state.png`

### Q7: What is the average property price by city?
- **Finding:** Tier-1 tech and financial capitals dominate aggregate valuations:
  - Top cities include Mumbai, Bangalore, Pune, Gurgaon, and Hyderabad with average listing valuations between ₹265L and ₹290L.
  - Tier-2 regional hubs (Jaipur, Indore, Lucknow) provide attractive yields for entry-level investors (average ₹210L–₹235L).
- **Visualization:** `reports/figures/eda_07_avg_price_by_city.png`

### Q8: What is the median age of properties by locality?
- **Finding:** Locality median property ages cluster evenly around **18.0 years**, reflecting a mature housing market balanced between new residential extensions (0–5 years) and established neighborhoods (>20 years).
- **Visualization:** `reports/figures/eda_08_median_age_by_locality.png`

### Q9: How is BHK distributed across cities?
- **Finding:** Across all major metro hubs, **2 BHK and 3 BHK configurations account for ~55%** of active inventory, representing the core demand of working families and institutional tenants. 4 and 5 BHK units are concentrated in suburban gated communities and villas.
- **Visualization:** `reports/figures/eda_09_bhk_distribution_across_cities.png`

### Q10: What are the price trends for the top 5 most expensive localities?
- **Finding:** The top 5 micro-markets maintain an average 40% valuation premium over their respective municipal medians. In these micro-markets, gated security and clubhouse facilities amplify valuations by an additional 15–20%.
- **Visualization:** `reports/figures/eda_10_top5_expensive_localities.png`

---

## Section 3: Feature Relationships & Correlations (Questions 11–15)

### Q11: How are numeric features correlated with each other?
- **Finding:** 
  - `Size_in_SqFt` vs `Price_in_Lakhs`: strong positive (r = 0.68)
  - `Infrastructure_Score` vs `Annual_Appreciation_Rate`: strong positive (r = 0.62)
  - `Age_of_Property` vs `Annual_Appreciation_Rate`: negative (r = -0.34), showing that older structures appreciate more slowly due to depreciation.
- **Visualization:** `reports/figures/eda_11_numeric_correlation_matrix.png`

### Q12: How do nearby schools relate to price per sq ft?
- **Finding:** Properties situated within high-density educational zones (7–10 schools within 3km) demonstrate a **12.5% price premium** and lower marketing turnover time.
- **Visualization:** `reports/figures/eda_12_schools_vs_price_per_sqft.png`

### Q13: How do nearby hospitals relate to price per sq ft?
- **Finding:** Proximity to healthcare facilities increases valuation steadily up to 6–7 hospitals. Beyond 8 facilities, marginal price appreciation flattens, indicating saturation.
- **Visualization:** `reports/figures/eda_13_hospitals_vs_price_per_sqft.png`

### Q14: How does price vary by furnished status?
- **Finding:** Fully Furnished properties average **₹262.4 Lakhs**, Semi-Furnished average **₹254.1 Lakhs**, and Unfurnished average **₹247.2 Lakhs**. Furnished assets offer investors turnkey readiness and immediate rental yields.
- **Visualization:** `reports/figures/eda_14_price_by_furnished_status.png`

### Q15: How does price per sq ft vary by property facing direction?
- **Finding:** East-facing and North-facing properties exhibit modest pricing premiums (~3–5%) over West and South orientations, reflecting cultural preferences and natural lighting optimization.
- **Visualization:** `reports/figures/eda_15_price_by_facing_direction.png`

---

## Section 4: Investment, Amenities & Ownership Analysis (Questions 16–20)

### Q16: How many properties belong to each owner type?
- **Finding:** Balanced distribution across market channels:
  - **Owner Sales:** ~33.4% (Direct resale liquidity)
  - **Builder Developments:** ~33.3% (Primary new market inventory)
  - **Broker Intermediations:** ~33.3% (Agency managed)
- **Visualization:** `reports/figures/eda_16_owner_type_distribution.png`

### Q17: How many properties are available under each availability status?
- **Finding:** **Ready-to-Move** (~50.1%) vs **Under-Construction** (~49.9%) splits inventory equally, allowing investors to choose between immediate rental cash flow vs pre-completion capital appreciation discounts.
- **Visualization:** `reports/figures/eda_17_availability_status.png`

### Q18: Does parking space affect property price?
- **Finding:** Dedicated parking creates a clear valuation premium (**₹260.8L with parking vs ₹248.3L without**), making it a non-negotiable amenity for middle- and luxury-segment buyers.
- **Visualization:** `reports/figures/eda_18_parking_vs_price.png`

### Q19: How do amenities affect price per sq ft?
- **Finding:** Price per sq.ft increases monotonically from 0 amenities to 5 amenities (Gym, Pool, Clubhouse, Garden, Playground). Properties with all 5 amenities command an average **18.2% valuation premium**.
- **Visualization:** `reports/figures/eda_19_amenities_vs_price_per_sqft.png`

### Q20: How does public transport accessibility relate to price per sq ft or investment potential?
- **Finding:** Properties with **High** public transport accessibility achieve a **58.4% Good Investment classification rate**, compared to only **14.2%** for Low accessibility areas. Metro and arterial bus connectivity represents the single most decisive factor for outperforming capital appreciation benchmarks.
- **Visualization:** `reports/figures/eda_20_transport_vs_investment.png`
