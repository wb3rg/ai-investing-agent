#!/usr/bin/env python3
"""
AI Investment Agent Framework - Practical Implementation Example
Using Agno framework with DeepSeek R1 model from OpenRouter

This example demonstrates a complete working implementation of the 
investment agent framework described in the guide.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Agno framework imports
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.yfinance import YFinanceTools
from agno.tools.financial_datasets import FinancialDatasetsTools
from agno.tools.openbb import OpenBBTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.python import PythonTools
from agno.teams import Team

# Configuration
class Config:
    """Configuration class for the investment agent framework"""
    
    # API Keys - Set these as environment variables
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
    
    # Model configuration
    MODEL_ID = "deepseek/deepseek-r1-0528:free"  # Free DeepSeek R1 model
    
    # Risk management parameters
    MAX_POSITION_SIZE = 0.10  # Maximum 10% position size
    MAX_PORTFOLIO_CONCENTRATION = 0.30  # Maximum 30% in any sector
    STOP_LOSS_PERCENTAGE = 0.15  # 15% stop loss
    
    # Performance thresholds
    MIN_SHARPE_RATIO = 1.0
    MAX_DRAWDOWN = 0.20  # 20% maximum drawdown
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        print("Configuration validated successfully")

# Initialize the DeepSeek R1 model
def create_model():
    """Create and configure the DeepSeek R1 model via OpenRouter"""
    return OpenRouter(
        id=Config.MODEL_ID,
        api_key=Config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=4096,
        temperature=0.7,  # Optimal for financial reasoning
        top_p=0.9
    )

class MarketDataAgent:
    """
    Level 1 Agent: Data Collection and Market Intelligence
    
    Responsibilities:
    - Collect real-time and historical market data
    - Aggregate news and analyst recommendations
    - Cross-validate data from multiple sources
    """
    
    def __init__(self, model):
        self.agent = Agent(
            name="Market Data Specialist",
            model=model,
            tools=[
                YFinanceTools(
                    stock_price=True,
                    company_info=True,
                    historical_prices=True,
                    company_news=True,
                    analyst_recommendations=True,
                    technical_indicators=True,
                    stock_fundamentals=True
                ),
                FinancialDatasetsTools(
                    enable_financial_statements=True,
                    enable_market_data=True,
                    enable_company_info=True,
                    enable_news=True
                ) if Config.FINANCIAL_DATASETS_API_KEY else None,
                OpenBBTools(),
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are a market data specialist focused on collecting comprehensive financial information",
                "Always cross-reference data from multiple sources for accuracy",
                "Format financial data in clear, structured tables using markdown",
                "Identify and flag any data inconsistencies or anomalies",
                "Provide data quality assessments and confidence scores",
                "Focus on both quantitative metrics and qualitative insights"
            ],
            markdown=True,
            show_tool_calls=True,
            memory_file="market_data_memory.json"
        )
    
    def collect_market_data(self, symbol: str) -> str:
        """Collect comprehensive market data for a given symbol"""
        query = f"""
        Collect comprehensive market data for {symbol}:
        
        1. Current stock price and basic information
        2. Historical price performance (1Y, 6M, 3M, 1M)
        3. Key financial metrics and ratios
        4. Recent news and developments (last 30 days)
        5. Analyst recommendations and price targets
        6. Trading volume and liquidity metrics
        7. Company profile and business description
        
        Cross-validate key metrics from multiple data sources and flag any inconsistencies.
        Present the data in clear, organized tables with confidence assessments.
        """
        
        return self.agent.run(query)

class TechnicalAnalysisAgent:
    """
    Level 2 Agent: Technical Analysis and Chart Pattern Recognition
    
    Responsibilities:
    - Perform technical indicator analysis
    - Identify chart patterns and trends
    - Generate trading signals
    - Assess technical momentum and sentiment
    """
    
    def __init__(self, model):
        self.agent = Agent(
            name="Technical Analysis Expert",
            model=model,
            tools=[
                YFinanceTools(
                    historical_prices=True,
                    technical_indicators=True
                ),
                PythonTools(),
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are a technical analysis expert specializing in chart analysis and trading signals",
                "Perform comprehensive technical analysis using multiple indicators",
                "Calculate and interpret: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, Williams %R",
                "Identify chart patterns: head and shoulders, triangles, flags, support/resistance",
                "Provide clear buy/sell/hold signals with confidence levels (1-10 scale)",
                "Always explain the reasoning behind technical conclusions",
                "Consider multiple timeframes for analysis (daily, weekly, monthly)",
                "Use Python code to calculate custom indicators when needed"
            ],
            markdown=True,
            memory_file="technical_analysis_memory.json"
        )
    
    def analyze_technical_indicators(self, symbol: str, period: str = "6mo") -> str:
        """Perform comprehensive technical analysis"""
        query = f"""
        Perform detailed technical analysis on {symbol} for the past {period}:
        
        1. **Trend Analysis**:
           - Overall trend direction (uptrend/downtrend/sideways)
           - Key support and resistance levels
           - Trend strength and momentum
        
        2. **Technical Indicators**:
           - Moving Averages (SMA 20, 50, 200 and EMA 12, 26)
           - RSI (14-period) with overbought/oversold levels
           - MACD with signal line crossovers
           - Bollinger Bands analysis
           - Stochastic Oscillator
        
        3. **Chart Patterns**:
           - Identify any significant chart patterns
           - Volume analysis and confirmation
           - Breakout or breakdown signals
        
        4. **Trading Signals**:
           - Generate clear buy/sell/hold recommendation
           - Confidence level (1-10 scale)
           - Entry price suggestions
           - Stop-loss and take-profit levels
           - Position sizing recommendations
        
        Use Python calculations where appropriate to verify indicator values.
        """
        
        return self.agent.run(query)

class FundamentalAnalysisAgent:
    """
    Level 2 Agent: Fundamental Analysis and Valuation
    
    Responsibilities:
    - Analyze financial statements and ratios
    - Perform valuation analysis (DCF, P/E, P/B, etc.)
    - Assess competitive position and industry trends
    - Evaluate management quality and corporate governance
    """
    
    def __init__(self, model):
        self.agent = Agent(
            name="Fundamental Analysis Expert",
            model=model,
            tools=[
                FinancialDatasetsTools(
                    enable_financial_statements=True,
                    enable_company_info=True,
                    enable_market_data=True
                ) if Config.FINANCIAL_DATASETS_API_KEY else None,
                YFinanceTools(
                    stock_fundamentals=True,
                    income_statements=True,
                    key_financial_ratios=True,
                    company_info=True
                ),
                PythonTools(),
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are a fundamental analysis expert focused on company valuation and financial health",
                "Conduct thorough fundamental analysis of companies and securities",
                "Analyze financial statements: Income Statement, Balance Sheet, Cash Flow",
                "Calculate and interpret key ratios: P/E, P/B, ROE, ROA, Debt/Equity, Current Ratio",
                "Perform valuation analysis using DCF, comparable company analysis, and asset-based methods",
                "Assess competitive advantages, moats, and industry position",
                "Evaluate management quality, corporate governance, and strategic direction",
                "Compare performance against industry peers and benchmarks",
                "Provide intrinsic value estimates with methodology explanation"
            ],
            markdown=True,
            memory_file="fundamental_analysis_memory.json"
        )
    
    def analyze_fundamentals(self, symbol: str) -> str:
        """Perform comprehensive fundamental analysis"""
        query = f"""
        Conduct comprehensive fundamental analysis for {symbol}:
        
        1. **Financial Health Assessment**:
           - Revenue growth trends (5-year and recent quarters)
           - Profitability metrics (gross, operating, net margins)
           - Cash flow analysis (operating, free cash flow)
           - Balance sheet strength (debt levels, liquidity ratios)
        
        2. **Valuation Metrics**:
           - Current valuation ratios (P/E, P/B, P/S, EV/EBITDA)
           - Historical valuation trends
           - Comparison with industry averages
           - PEG ratio and growth-adjusted metrics
        
        3. **Competitive Analysis**:
           - Market position and competitive advantages
           - Industry trends and growth prospects
           - Regulatory environment and risks
           - Peer comparison analysis
        
        4. **Management and Governance**:
           - Management track record and strategy
           - Corporate governance quality
           - Capital allocation decisions
           - Insider ownership and alignment
        
        5. **Intrinsic Value Calculation**:
           - DCF model with assumptions
           - Multiple valuation approaches
           - Fair value range estimation
           - Margin of safety assessment
        
        Provide clear investment thesis with upside/downside scenarios.
        """
        
        return self.agent.run(query)

class RiskManagementAgent:
    """
    Level 3 Agent: Risk Assessment and Management
    
    Responsibilities:
    - Calculate portfolio risk metrics (VaR, CVaR, Sharpe ratio)
    - Assess correlation and concentration risks
    - Implement position sizing recommendations
    - Monitor and manage downside protection
    """
    
    def __init__(self, model):
        self.agent = Agent(
            name="Risk Management Specialist",
            model=model,
            tools=[
                PythonTools(),
                YFinanceTools(historical_prices=True),
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are a risk management specialist focused on protecting capital and optimizing risk-adjusted returns",
                "Calculate comprehensive risk metrics: VaR, CVaR, maximum drawdown, Sharpe ratio, Sortino ratio",
                "Assess portfolio concentration risks and correlation analysis",
                "Implement position sizing using Kelly Criterion and risk budgeting",
                "Monitor portfolio stress testing under various market scenarios",
                "Provide dynamic hedging strategies and downside protection",
                "Use Python for advanced risk calculations and monte carlo simulations",
                "Always consider tail risks and black swan events"
            ],
            markdown=True,
            memory_file="risk_management_memory.json"
        )
    
    def assess_investment_risk(self, symbol: str, portfolio_weight: float = 0.05) -> str:
        """Assess risk for individual investment and portfolio impact"""
        query = f"""
        Conduct comprehensive risk assessment for {symbol} with proposed portfolio weight of {portfolio_weight:.1%}:
        
        1. **Individual Security Risk**:
           - Historical volatility analysis (daily, monthly returns)
           - Maximum drawdown periods and recovery times
           - Beta analysis vs market benchmark
           - Downside deviation and upside capture
        
        2. **Risk Metrics Calculation**:
           - 1-day and 10-day Value at Risk (95% and 99% confidence)
           - Conditional Value at Risk (Expected Shortfall)
           - Sharpe and Sortino ratios
           - Calmar ratio (return/max drawdown)
        
        3. **Portfolio Impact Analysis**:
           - Correlation with major market indices
           - Sector and style exposure
           - Liquidity risk assessment
           - Concentration risk implications
        
        4. **Scenario Analysis**:
           - Stress testing under market crash scenarios (-20%, -40%)
           - Interest rate sensitivity analysis
           - Recession probability impact
           - Currency and commodity exposure
        
        5. **Position Sizing Recommendation**:
           - Kelly Criterion optimal sizing
           - Risk budgeting allocation
           - Maximum position size given risk constraints
           - Stop-loss and profit-taking levels
        
        Use Python calculations for risk metrics and provide specific risk management recommendations.
        """
        
        return self.agent.run(query)

class PortfolioManagerAgent:
    """
    Level 4 Agent: Portfolio Construction and Optimization
    
    Responsibilities:
    - Optimize asset allocation using Modern Portfolio Theory
    - Implement portfolio rebalancing strategies
    - Monitor portfolio performance and attribution
    - Coordinate multi-asset portfolio decisions
    """
    
    def __init__(self, model):
        self.agent = Agent(
            name="Portfolio Manager",
            model=model,
            tools=[
                PythonTools(),
                YFinanceTools(),
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are a portfolio manager responsible for optimal asset allocation and portfolio construction",
                "Apply Modern Portfolio Theory and risk management principles",
                "Optimize portfolios for risk-adjusted returns using mean-variance optimization",
                "Consider diversification across assets, sectors, and geographies",
                "Implement strategic and tactical asset allocation decisions",
                "Monitor portfolio performance attribution and rebalancing needs",
                "Use Python for portfolio optimization calculations and backtesting",
                "Balance theoretical optimization with practical implementation constraints"
            ],
            markdown=True,
            memory_file="portfolio_memory.json"
        )
    
    def optimize_portfolio_allocation(self, holdings: Dict[str, float], target_return: float = 0.12) -> str:
        """Optimize portfolio allocation using MPT principles"""
        query = f"""
        Optimize portfolio allocation for current holdings with target annual return of {target_return:.1%}:
        
        Current Holdings: {holdings}
        
        1. **Portfolio Analysis**:
           - Current portfolio composition and weights
           - Risk-return characteristics of each holding
           - Correlation matrix analysis
           - Diversification effectiveness
        
        2. **Optimization Process**:
           - Mean-variance optimization for efficient frontier
           - Sharpe ratio maximization
           - Risk budgeting across positions
           - Constraint implementation (max weights, sectors)
        
        3. **Rebalancing Recommendations**:
           - Optimal target weights vs current allocation
           - Trading recommendations (buy/sell/hold)
           - Transaction cost considerations
           - Tax efficiency implications
        
        4. **Performance Projections**:
           - Expected portfolio return and volatility
           - Sharpe ratio and risk-adjusted metrics
           - Downside risk and maximum drawdown estimates
           - Monte Carlo simulation results
        
        Use Python code to implement portfolio optimization algorithms and provide specific allocation recommendations.
        """
        
        return self.agent.run(query)

class InvestmentOrchestrator:
    """
    Level 5 Agent: Master Investment Decision Orchestrator
    
    This is the highest-level agent that coordinates all specialist agents
    and makes final investment decisions using DeepSeek R1's advanced reasoning.
    """
    
    def __init__(self, model):
        # Initialize all specialist agents
        self.market_agent = MarketDataAgent(model)
        self.technical_agent = TechnicalAnalysisAgent(model)
        self.fundamental_agent = FundamentalAnalysisAgent(model)
        self.risk_agent = RiskManagementAgent(model)
        self.portfolio_agent = PortfolioManagerAgent(model)
        
        # Master orchestrator with DeepSeek R1's reasoning capabilities
        self.orchestrator = Agent(
            name="Investment Decision Orchestrator",
            model=model,
            tools=[
                ReasoningTools(add_instructions=True)
            ],
            instructions=[
                "You are the master investment decision maker using advanced reasoning capabilities",
                "Synthesize comprehensive analysis from all specialist agents",
                "Apply systematic investment decision-making framework",
                "Consider multiple scenarios, probabilities, and outcomes",
                "Make final investment recommendations based on risk-adjusted expected returns",
                "Provide detailed reasoning chains for all investment decisions",
                "Consider behavioral finance principles and market psychology",
                "Always quantify confidence levels and provide sensitivity analysis",
                "Implement systematic review and learning from past decisions"
            ],
            markdown=True,
            memory_file="orchestrator_memory.json"
        )
    
    async def comprehensive_investment_analysis(self, symbol: str, portfolio_context: Optional[Dict] = None) -> Dict:
        """
        Orchestrate comprehensive investment analysis workflow
        
        This method coordinates all specialist agents and provides final investment recommendation
        """
        analysis_results = {}
        
        try:
            print(f"\n🔍 Starting comprehensive investment analysis for {symbol}")
            
            # Step 1: Market Data Collection
            print("📊 Collecting market data...")
            analysis_results['market_data'] = self.market_agent.collect_market_data(symbol)
            
            # Step 2: Technical Analysis
            print("📈 Performing technical analysis...")
            analysis_results['technical_analysis'] = self.technical_agent.analyze_technical_indicators(symbol)
            
            # Step 3: Fundamental Analysis
            print("📋 Conducting fundamental analysis...")
            analysis_results['fundamental_analysis'] = self.fundamental_agent.analyze_fundamentals(symbol)
            
            # Step 4: Risk Assessment
            print("⚠️ Assessing investment risks...")
            analysis_results['risk_assessment'] = self.risk_agent.assess_investment_risk(symbol)
            
            # Step 5: Portfolio Impact (if portfolio context provided)
            if portfolio_context:
                print("💼 Analyzing portfolio impact...")
                analysis_results['portfolio_impact'] = self.portfolio_agent.optimize_portfolio_allocation(portfolio_context)
            
            # Step 6: Master Synthesis and Decision
            print("🧠 Synthesizing final investment decision...")
            synthesis_query = self._create_synthesis_query(symbol, analysis_results, portfolio_context)
            analysis_results['final_recommendation'] = self.orchestrator.run(synthesis_query)
            
            print(f"✅ Analysis complete for {symbol}")
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error in analysis for {symbol}: {str(e)}")
            return {"error": str(e), "symbol": symbol}
    
    def _create_synthesis_query(self, symbol: str, analysis_results: Dict, portfolio_context: Optional[Dict]) -> str:
        """Create comprehensive synthesis query for the orchestrator"""
        
        portfolio_info = f"\nPortfolio Context: {portfolio_context}" if portfolio_context else ""
        
        return f"""
        Based on the comprehensive analysis below, provide a detailed investment recommendation for {symbol}:
        
        ## Market Data Analysis
        {analysis_results.get('market_data', 'Not available')}
        
        ## Technical Analysis
        {analysis_results.get('technical_analysis', 'Not available')}
        
        ## Fundamental Analysis
        {analysis_results.get('fundamental_analysis', 'Not available')}
        
        ## Risk Assessment
        {analysis_results.get('risk_assessment', 'Not available')}
        {portfolio_info}
        
        ## Required Final Recommendation Format
        
        Please provide a comprehensive investment decision with the following structure:
        
        ### 1. Executive Summary
        - **Investment Decision**: [BUY/HOLD/SELL]
        - **Confidence Level**: [1-10 scale with reasoning]
        - **Investment Horizon**: [Short/Medium/Long-term]
        - **Risk Rating**: [Low/Medium/High with explanation]
        
        ### 2. Investment Thesis
        - Primary investment drivers and catalysts
        - Key competitive advantages or concerns
        - Market opportunity and growth prospects
        - Risk factors and mitigation strategies
        
        ### 3. Valuation and Targets
        - **Target Price**: [12-month price target with range]
        - **Stop Loss**: [Recommended stop-loss level]
        - **Expected Return**: [Risk-adjusted expected return]
        - **Valuation Method**: [Primary valuation approach used]
        
        ### 4. Position Sizing and Risk Management
        - **Recommended Position Size**: [% of portfolio]
        - **Maximum Position Size**: [Risk-adjusted maximum]
        - **Entry Strategy**: [Timing and approach]
        - **Exit Strategy**: [Profit-taking and stop-loss plan]
        
        ### 5. Scenario Analysis
        - **Base Case** (60% probability): Expected outcome
        - **Bull Case** (20% probability): Best-case scenario
        - **Bear Case** (20% probability): Worst-case scenario
        
        ### 6. Key Monitoring Points
        - Critical metrics to track going forward
        - Catalysts that could change the thesis
        - Review schedule and decision checkpoints
        
        ### 7. Decision Reasoning Chain
        Provide step-by-step reasoning that led to this recommendation, including:
        - How you weighted different analysis factors
        - Key trade-offs and uncertainties considered
        - Potential biases and how you addressed them
        
        Use advanced reasoning to ensure the recommendation is well-justified and accounts for multiple scenarios and uncertainties.
        """

# Example usage and testing functions
class InvestmentFrameworkDemo:
    """
    Demonstration class showing how to use the investment framework
    """
    
    def __init__(self):
        # Validate configuration
        Config.validate_config()
        
        # Initialize model
        self.model = create_model()
        
        # Initialize orchestrator
        self.orchestrator = InvestmentOrchestrator(self.model)
    
    async def demo_single_stock_analysis(self, symbol: str = "AAPL"):
        """Demonstrate single stock analysis"""
        print(f"\n{'='*60}")
        print(f"🚀 AI Investment Agent Framework Demo")
        print(f"📊 Analyzing: {symbol}")
        print(f"🤖 Model: {Config.MODEL_ID}")
        print(f"{'='*60}")
        
        # Run comprehensive analysis
        results = await self.orchestrator.comprehensive_investment_analysis(symbol)
        
        if 'error' not in results:
            print(f"\n📄 Final Investment Recommendation for {symbol}:")
            print("="*60)
            print(results['final_recommendation'])
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"investment_analysis_{symbol}_{timestamp}.json"
            
            # Convert results to JSON-serializable format
            json_results = {
                "symbol": symbol,
                "timestamp": timestamp,
                "analysis_summary": {
                    "market_data_length": len(results.get('market_data', '')),
                    "technical_analysis_length": len(results.get('technical_analysis', '')),
                    "fundamental_analysis_length": len(results.get('fundamental_analysis', '')),
                    "risk_assessment_length": len(results.get('risk_assessment', '')),
                    "final_recommendation_length": len(results.get('final_recommendation', ''))
                },
                "final_recommendation": results.get('final_recommendation', '')
            }
            
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n💾 Results saved to: {filename}")
            
        else:
            print(f"\n❌ Analysis failed: {results['error']}")
    
    async def demo_portfolio_analysis(self):
        """Demonstrate portfolio analysis with multiple holdings"""
        portfolio = {
            "AAPL": 0.20,  # 20% Apple
            "MSFT": 0.15,  # 15% Microsoft
            "GOOGL": 0.10, # 10% Google
            "TSLA": 0.08,  # 8% Tesla
            "NVDA": 0.12   # 12% Nvidia
        }
        
        print(f"\n{'='*60}")
        print(f"💼 Portfolio Analysis Demo")
        print(f"Current Holdings: {portfolio}")
        print(f"{'='*60}")
        
        # Analyze each holding
        for symbol, weight in portfolio.items():
            print(f"\n🔍 Analyzing {symbol} (Current weight: {weight:.1%})")
            results = await self.orchestrator.comprehensive_investment_analysis(
                symbol, 
                portfolio_context=portfolio
            )
            
            if 'error' not in results:
                print(f"✅ {symbol} analysis complete")
            else:
                print(f"❌ {symbol} analysis failed: {results['error']}")

# Main execution
async def main():
    """Main execution function"""
    try:
        # Initialize demo
        demo = InvestmentFrameworkDemo()
        
        # Run single stock analysis
        await demo.demo_single_stock_analysis("AAPL")
        
        # Uncomment to run portfolio analysis
        # await demo.demo_portfolio_analysis()
        
    except Exception as e:
        print(f"❌ Framework error: {str(e)}")
        print("Please check your configuration and API keys")

if __name__ == "__main__":
    print("🤖 AI Investment Agent Framework")
    print("Using Agno + DeepSeek R1 for Intelligent Investment Analysis")
    print("\nRequired Environment Variables:")
    print("- OPENROUTER_API_KEY (required)")
    print("- FINANCIAL_DATASETS_API_KEY (optional)")
    print("\nStarting analysis...")
    
    # Run the demo
    asyncio.run(main())
