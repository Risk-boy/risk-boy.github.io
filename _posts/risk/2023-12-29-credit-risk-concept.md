---

title: "Theory of Credit Risk Model"

categories:

- RSIK

toc: true
toc_sticky: true

date: 2023-12-29
last_modified_at: 2023-12-29

## Risk

**Consequences of Uncertainty**

### Dimensions of Risk

- Event: what has to happen for the Risk to occur
- Duration: for how long are we exposed to this Risk
- Frequency: Probability or Rate of the Risk event occuring
- Severity: Magnitude of the consequences
- Correlation: Relationship with other Risk Events
- Capital: Reseves needed to support Risk

### Uncertainty to each dimension

- Event: The cause could be uncertain
- Duration: The exposure could be uncertain
- Frequency: The true prob / rate could be uncertain
- Severity: The size of consequence could be uncertain
- Correlation: Relationships could be uncertain and subject to change
- Capital: Reserves might only give a level of confidence and not certainty

### For Example) Life Insurance

- Exclusions to control for event uncertainty
- Term Assurance to control for duration uncertainty
- Actuarial Modeling to try reduce rate uncertainty
- Fixed Benefits to control severity uncertainty
- Sell Life Annuities to get diversification and offsetting benefits
- Pool multiple similar risks to increase capital confidence(Flip coin)

### STANDARD APPROACH

1. Observe a population of size n for duration T

2. Count number of risk events that occur over duration T

   $$Frequency = \frac{\#\ of\ risk\ events}{n} $$

3. Observe size of loss from each risk event over duration T

   $$Severity = \frac{\sum\\of\ all\ losses}{\#\ of\ risk\ events}$$

   $$Risk\ Price=Severity\times Frequncy$$

### WHY MEASURE RISK

- Measuring Risk helps us to Manage it efficiently

- low frequency & low severity: Retain
- low frequency & high severity: Transfer
- high frequency & low severity: Control
- high frequency & high severity: Avoid

1. Do I want to take this Risk?
2. How much should I pay to transfer it? What is a fair premium?
3. What should my internal control budget be?
4. How much capital do I need to retain this Risk?

### Mathematical Properties of Risk Measures

- **Coherent**: logical and consistenct

- **Coherent Axioms**

  - Monotonicity: $if\ x_1 \lt x_2\ than\ f(x_1) \lt f(x_2)$

  - Subadditivity: $f(x_1+x_2) \le f(x_1)+f(x_2)$

  - Positive Homogeneity: $f(kx)=kf(x)\ (k\ is\ a\ positive\ constant)$

  - Translation Invariance: $f(k+x)=k+f(x)\ (k\ is\ a\ constant)$

- **Convex**:

​ $$f(tx_1 + (1-t)x_2) \le tf(x_1)+(1-t)f(x_2)\ (0\le t \le 1)$$

- What does this mean?
  - Merge of Risks decreases Risk Profile
  - Diversification
  - Business units must stay within limits
  - But what about Concentration & Systemic Risk

### Credit Risk

- Event: 3rd Party fails to meet their obligation
- Duration: Specified by contract / type of instrument
- Frequency: Very low, normally only happens once(default)
- Severity: Very high, due to recovery, amount is uncertain
- Correlation: Low in stable times, High in distressed times(Economic conditions)
- Capital: None is isolation, but if using others' money then reserves need to be established. Basel Accords

### Identifying Credit Risk

1. Lending Money to someone who then cannot repay
   - Secured: Collateral
   - Unsecured: No Collateral
2. Buying Public Debt in form of Corporate or Government Bond
   - Credit Spread Deteriorate(increases) -> price decreases
   - Credit Spread Improve(decreases) -> price increases
3. Counterparty Risk
   - Insurance
   - Derivatives
   - Banking(Financial Institution) -> often not considered because protected by bailout
   - Outsourcing Functions

### Overview of Models

1. Lending
   1. Face to Face Model(qualitative)
   2. Credit Score Model(quantitative)
   3. Due-Diligence Model
2. Public Debt
   1. Merton Model
   2. KMV Model
   3. Two-State Model
   4. Jarrow-Lando-Turnbull Model
3. Other Models
   1. Interest Rate Models: Frequency
   2. Value at Risk: Capital & Severity
   3. Copula's: Correlation

### Credit Modelling Challenges

1. Lack of Data
   - Bank don't share their experience
2. Since Frequency is Low, Severity Data will be lacking even more(Hard to fit a distribution)

3. Bond Durations
   - Not all corporates have traded debt
   - Those that do have differenct durations
   - Different Model for 10yrs and 20yrs
4. Credit Enhancements
   - Guarantees by another party -> FRAUD
   - Derivatives CDS, CDO -> ABUSE
5. Changing Correlation
   - Low when Stable Economy
   - High during Recession
6. Model mixing
   - Differenct technique for Frequency & Severity
   - Complexity joining and model risk
7. Credit Rating Agency
   - Conflicting information
   - Business could game the system
   - Conflict of Interest Risks

### Face to Face Model

1. Personal Meeting iwth Bank Manager
2. Questions & Requirements

   1. Security: Collateral or Surety
   2. Borrower: Degree, Job, Age, Record(Credit History, Criminal)
   3. Purpose: How will loan be used
   4. Financial Ratio: Cashflows, Assets, Liabilities
   5. Ecomony: Business Confidence, GDP, interests rates

3. Decision to Approve / Reject
4. Terms & Conditions
   1. Duration
   2. Interest Rate
5. Risk to assess
   1. Default
   2. Recover
   3. Change
6. Private Debt / Individual / Small Business
7. Big Business: Team with more Questions & Requirements

### Credit Score Model

1. Automated Version of Face to Face model
2. Act as a filter to save time
3. Make decision to process loans fast
4. Requires Data & Expertise to set Rules
5. Has blind spots, can't consider everything

6. Weighting to each factor is subjective

### Introduction to Derivatives

- Derive their value from another underlying asset
- Purpose is to manage Market Risk
  - Hedger: use to decrease Market Risk
  - Speculator: use to increase Market Risk(Bet)
- Hedgers Example

  - Farmer wants to lock in price of sale
  - Restaurant want to lock in price of costs

- Speculators Example

  - Asset manager A thinks oil price will go up
  - Asset manager B thinks oil price will go down

- Over the Counter

  - Between two parties
  - Customisable
  - Counter Party Risk
  - Difficult to close out early

- Exchange Trade

  - Both Parties go through a Clearing House
  - Standardised
  - Secure
  - Easy to close out early

- Types

  - Futures, Forwards, Options, Swaps, Combinations, Exotics

- Assumptions

  - Complete Market

    - Everything has a price
    - Transaction costs are zero
    - Everyone has access to perfect information

  - No Arbitrage
    - All assets are priced correctly
    - Impossible to make a Risk Free Profit

### Futures & Forwards Contracts

- Future Delivery Price (K)

- Future Delivery Date (T)

- Value of Future at Delivery Date ($F_T$)

  - Long: $S_T-K$ (The buyer)

  - Short: $K-S_T$ (The seller)

#### Sub-Martingale

- $$F_T=S_T-K \ and \ E[S_T]=S_0e^{rT}\ (r\ is\ risk\ free\ rate)$$
- $E[F_T]=0 \ (since\ F_0=0)$

- $K=S_0e^{rT}$

- Expected future price is current price adjusted for T.V.M(Time Value of Money)
- If $K>S_0e^{rT}$ Long Asset and Short Future
- If $K < S_0e^{rT}$ Short Asset and Long Future

### Call and Put Options

- Long Call

  - **right, not obligation**, to **Buy asset** at a specific Price at a specific Date

- Long Put
  - **right, not obligation**, to **Sell asset** at a specific Price at a specific Date
- For both a Premium needs to be paid to the Short Party
  - Short Option: Receive Premium and Obligation
  - Long Option: Pay Premium and receive a Right
- Limit Downside Risk
- Maintain Upside Risk

#### Black Scholes Formula

- $C_0 = S_0N(d_1)-Ke^{-rT}N(d_2)$
- $P_0=Ke^{-rT}N(-d_2)-S_0N(-d_1)$

- $d_1= \frac{ln(S_0/K)+(r+\frac{\sigma^{2}}{2})T}{\sigma \sqrt{T}}$

- $d_2= \frac{ln(S_0/K)+(r-\frac{\sigma^{2}}{2})T}{\sigma \sqrt{T}}= d_1-\sigma\sqrt{T}$

### Factors affecting Option Prices

1. Share Price: $as \ S_0\uparrow,\ C\uparrow\ P\downarrow$

2. Strike Price: $as \ K\uparrow,\ C\downarrow\ P\uparrow$
