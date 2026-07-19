from pydantic import BaseModel,Field
from typing import List,Annotated, Optional, Dict

class PortfolioInput(BaseModel):
    portfolio_size:Annotated[float,Field(...,gt=0,lt=1000000,description='total value of the portfolio')]
    tickers : Annotated[List[str],Field(...,description='tickers of the stocks in the portfolio')]
    num_simulations : Annotated[int,Field(...,gt=999,lt=100001,description='number of predicted futures')]
    years: Annotated[int,Field(...,gt=0,lt=6,description= 'number of years worth of data to be considered')]
    weights: Annotated[Optional[List[float]],Field(default=None,description='portfolio weight per ticker, must match tickers length and sum to ~1.0. Omit for equal weighting.')]

class RiskMetricsOutput(BaseModel):
    var_95: Annotated[float, Field(description="The 95% Value at Risk dollar amount")]
    var_99: Annotated[float, Field(description="The 99% Value at Risk dollar amount")]
    cvar: Annotated[float, Field(description="Expected Shortfall (Average of worst 5% of outcomes)")]
    message: Annotated[str, Field(description="Status message of the computation")]


class NeuralNetTickerBreakdown(BaseModel):
    ticker: str
    as_of_date: str
    q05_return: Annotated[float, Field(description="Predicted 5% quantile next day log return for this ticker")]
    q01_return: Annotated[float, Field(description="Predicted 1% quantile next day log return for this ticker")]
    weight: float
    dollar_value: float
    var_95: Annotated[float, Field(description="Standalone dollar VaR at 95% for this ticker alone.")]
    var_99: Annotated[float, Field(description="Standalone dollar VaR at 99% for this ticker alone.")]


class NeuralNetResult(BaseModel):
    var_95: Annotated[float, Field(description="Portfolio-level dollar VaR at 95%, summed across tickers")]
    var_99: Annotated[float, Field(description="Portfolio-level dollar VaR at 99%, summed across tickers")]
    per_ticker: Dict[str, NeuralNetTickerBreakdown]
    message: Annotated[str, Field(description="Explains the correlation assumption behind this estimate")]


class CombinedRiskMetricsOutput(BaseModel):
    monte_carlo: RiskMetricsOutput
    neural_network: NeuralNetResult
    comparison_note: Annotated[str, Field(description="How to interpret differences between the two methods' numbers")]