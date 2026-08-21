# AU Retail Pricing & Margin Decision Lab

A synthetic Australian retail pricing analytics project that
demonstrates how cost movements, competitor pricing, price elasticity
and commercial guardrails can be combined into actionable SKU-level
pricing recommendations.

The project moves beyond descriptive pricing reporting to answer a more
practical commercial question:

> **Where should a retailer increase, hold or reduce price, and what is
> the expected impact on demand, sales and margin?**

The solution includes a multi-page Streamlit decision application, a
synthetic two-year pricing history, hierarchical elasticity estimation,
commercial calibration, scenario modelling and a prioritised
recommendation engine.

------------------------------------------------------------------------

## Business Problem

Retail pricing decisions require balancing several competing objectives:

-   protect gross margin when product costs increase
-   remain appropriately positioned against competitors
-   understand expected customer demand response to price changes
-   identify where price investment may generate incremental volume
-   avoid unnecessary price changes where evidence is weak
-   prioritise thousands of SKU-level decisions into an actionable queue

This project simulates that decision environment across **1,500
synthetic SKUs** in an Australian retail portfolio.

Rather than optimising price purely from a statistical model, the
decision framework combines:

**Cost → Price → Competitive Position → Elasticity → Demand → Margin →
Decision**

------------------------------------------------------------------------

## Streamlit Decision Lab

The application contains five connected decision views.

### 1. Executive Overview

Provides a portfolio-level view of the pricing opportunity, including:

-   28-day margin opportunity
-   expected sales and unit impact
-   current versus recommended price position
-   current versus recommended margin
-   recommended action mix
-   competitive price-position distribution
-   top margin opportunities
-   competitive exposure

### 2. Recommendation Queue

Prioritises SKU-level pricing decisions using expected commercial
impact, competitive position and decision confidence.

Recommendations are classified as:

-   **Increase Price**
-   **Hold Price**
-   **Reduce Price**
-   **Review**

The queue includes recommendation rationale, expected sales impact,
expected margin impact and decision confidence, and can be exported for
further analysis.

### 3. Scenario Explorer

Allows individual SKUs to be tested across alternative price points.

For each scenario the application estimates:

-   expected demand response
-   28-day units
-   28-day sales
-   28-day gross margin
-   margin percentage
-   competitive price index

This makes the trade-off between price, volume, competitiveness and
margin visible before a pricing decision is made.

### 4. Margin & Competition

Connects historical pricing behaviour with the current commercial
position.

The page includes:

-   cost pass-through analysis
-   cost, sell-price and competitor-price history
-   gross-margin trend
-   competitive price-index trend
-   portfolio price-position versus margin
-   most price-sensitive product classes
-   commercial watchlist

The watchlist highlights SKUs with margin pressure, competitive exposure
or recommended price investment requiring attention.

### 5. Model Diagnostics

Makes the modelling and evidence behind the recommendation engine
transparent.

Diagnostics include:

-   decision-model coverage
-   SKU-level versus benchmark evidence
-   decision-confidence distribution
-   elasticity evidence tiers
-   raw versus calibrated elasticity
-   calibrated elasticity distribution
-   SKU estimation quality
-   product-class model diagnostics
-   model-review queue

## Application Screenshots

### Executive Overview

![Executive Overview](docs/images/01_executive_overview.png)

### Recommendation Queue

![Recommendation Queue](docs/images/02_recommendation_queue.png)

### Scenario Explorer

![Scenario Explorer](docs/images/03_scenario_explorer.png)

### Margin & Competition

![Margin & Competition](docs/images/04_margin_competition.png)

### Model Diagnostics

![Model Diagnostics](docs/images/05_model_diagnostics.png)

------------------------------------------------------------------------

## Pricing Decision Framework

The recommendation engine does not rely on a single elasticity estimate.

It uses a hierarchical approach.

### 1. SKU Evidence

SKUs with sufficient historical price variation are eligible for
SKU-level elasticity estimation.

### 2. Product-Class Benchmark

Product-class estimates provide a broader pricing-response benchmark
when individual SKU evidence is limited or noisy.

### 3. Decision Hierarchy

SKU and product-class signals are combined according to evidence
strength.

Where SKU-level evidence is insufficient, the framework falls back to
the broader benchmark.

### 4. Commercial Calibration

Raw elasticity estimates are blended with commercial priors.

Model weight is determined by evidence confidence, reducing the
influence of unstable or extreme statistical estimates.

### 5. Pricing Decision Engine

The calibrated elasticity signal is combined with:

-   current cost
-   current sell price
-   competitor price
-   gross margin
-   competitive price position
-   expected demand response
-   commercial pricing guardrails

to generate the final recommendation.

------------------------------------------------------------------------

## Synthetic Data

The project uses fully synthetic data designed to resemble a
multi-category Australian retailer.

The dataset contains:

-   **1,500 SKUs**
-   **730 days of daily history per SKU**
-   **1,095,000 SKU-day observations**
-   three retail departments
-   multiple categories and product classes
-   product costs
-   regular sell prices
-   competitor prices
-   price changes
-   demand response
-   units sold
-   sales
-   gross margin
-   hidden true elasticity for model validation

The historical period spans:

**1 July 2024 to 30 June 2026**

No proprietary retailer data is used.

------------------------------------------------------------------------

## Elasticity Modelling

Price elasticity is estimated from historical price and demand
behaviour.

Because SKU-level estimates can become unstable when price variation is
limited, the project evaluates evidence strength before allowing
SKU-level estimates to influence pricing decisions.

The framework uses:

-   SKU-level elasticity estimation
-   product-class benchmarks
-   evidence tiers
-   model confidence
-   hierarchical fallback
-   commercial calibration

The synthetic dataset also contains hidden true elasticity values,
allowing direct comparison between estimated and known elasticity during
model development.

In a real retail environment, model quality would instead be evaluated
through holdout performance, controlled pricing experiments, stability
monitoring and realised pricing outcomes.

------------------------------------------------------------------------

## Commercial Outputs

The decision engine converts modelling outputs into commercially
interpretable measures including:

-   recommended price
-   recommended price change %
-   expected unit impact
-   expected sales impact
-   expected gross-margin impact
-   recommended competitive price index
-   decision confidence
-   recommendation rationale

Commercial impacts are presented over a **28-day scenario horizon**.

------------------------------------------------------------------------

## Project Structure

``` text
au-retail-pricing-margin-decision-lab/
│
├── app.py
├── app_pages/
│   ├── 1_Executive_Overview.py
│   ├── 2_Recommendation_Queue.py
│   ├── 3_Scenario_Explorer.py
│   ├── 4_Margin_Competition.py
│   └── 5_Model_Diagnostics.py
│
├── data/
│   ├── generated/
│   ├── runtime/
│   └── sample/
│
├── docs/
├── notebooks/
├── outputs/
│
├── src/
│   ├── app/
│   ├── data_generation/
│   ├── decision_engine/
│   ├── modelling/
│   ├── pricing/
│   └── qa/
│
├── tests/
├── pytest.ini
└── README.md
```

------------------------------------------------------------------------

## Quality Assurance

Automated tests validate core pricing and demand logic including:

-   competitor-price generation
-   price-index calculations
-   competitor price-gap calculations
-   daily demand integrity
-   sales calculations
-   gross-margin calculations
-   demand response to own-price changes
-   pricing recommendation guardrails
-   valid recommendation actions
-   one recommendation per SKU

Current test suite:

**20 tests passed**

Run the tests with:

``` bash
pytest -v
```

------------------------------------------------------------------------

## Running the Application

Create and activate the project environment, install the required
dependencies and launch Streamlit from the project root.

``` bash
streamlit run app.py
```

The application will open locally in your browser.

------------------------------------------------------------------------

## Technology Stack

-   Python
-   pandas
-   NumPy
-   statsmodels
-   Streamlit
-   Altair
-   PyArrow / Parquet
-   pytest
-   Git / GitHub

------------------------------------------------------------------------

## Key Skills Demonstrated

This project demonstrates an end-to-end approach to retail pricing
analytics across:

-   commercial problem framing
-   synthetic data engineering
-   pricing and margin analytics
-   competitor-price analysis
-   price elasticity modelling
-   model calibration
-   scenario simulation
-   decision-engine design
-   commercial guardrails
-   recommendation prioritisation
-   model diagnostics
-   interactive analytics application development
-   automated testing

------------------------------------------------------------------------

## Disclaimer

This is a portfolio project built using synthetic data.

The retailer, SKUs, prices, demand behaviour, competitor prices and
commercial outcomes are simulated and do not represent confidential or
proprietary information from any real retailer.

The project is intended to demonstrate analytical methodology,
commercial reasoning and decision-support application design rather than
provide production pricing recommendations.
