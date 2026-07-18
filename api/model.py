from pydantic import BaseModel,Field
from typing import List,Annotated

class PortfolioInput(BaseModel):
    portfolio_size:Annotated[float,Field(...,gt=0,lt=1000000,description='total value of the portfolio')]
    tickers : Annotated[List[str],Field(...,description='tickers of the stocks in the portfolio')]
    num_simulations : Annotated[int,Field(...,gt=999,lt=100001,description='number of predicted futures')]
    years: Annotated[int,Field(...,gt=0,lt=6,description= 'number of years worth of data to be considered')]

class RiskMetricsOutput(BaseModel):
    var_95: Annotated[float, Field(description="The 95% Value at Risk dollar amount")]
    var_99: Annotated[float, Field(description="The 99% Value at Risk dollar amount")]
    cvar: Annotated[float, Field(description="Expected Shortfall (Average of worst 5% of outcomes)")]
    message: Annotated[str, Field(description="Status message of the computation")]



