# AI Investment Agent Framework

A comprehensive Python framework for building intelligent investment agents using **Agno** (multi-agent systems) and **DeepSeek R1** (advanced reasoning model).

## 🎯 Overview

This framework demonstrates how to build sophisticated AI investment agents capable of:

- **Multi-source financial data collection** (YFinance, Financial Datasets API, OpenBB)
- **Advanced technical analysis** with multiple indicators and chart patterns
- **Comprehensive fundamental analysis** including DCF valuation and peer comparison
- **Sophisticated risk management** with VaR, portfolio optimization, and position sizing
- **Multi-agent collaboration** with specialized roles and shared reasoning
- **Production-ready deployment** with FastAPI and monitoring capabilities

## 🏗️ Architecture

The framework implements a hierarchical multi-agent architecture:

```
┌─────────────────────────────────────────────────────────┐
│                 AI Investing Agent Framework             │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Market    │  │ Technical   │  │ Fundamental │      │
│  │ Data Agent  │  │ Analysis    │  │ Analysis    │      │
│  │             │  │   Agent     │  │   Agent     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Portfolio   │  │   Risk      │  │ Execution   │      │
│  │ Manager     │  │ Management  │  │   Agent     │      │
│  │   Agent     │  │   Agent     │  │             │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          Master Orchestrator Agent                  │ │
│  │     (DeepSeek R1 for Complex Reasoning)            │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the framework files
# Run the setup script
python setup.py

# Or install manually
pip install -r requirements.txt
```

### 2. Configuration

Edit the `.env` file with your API keys:

```bash
# Required: OpenRouter API key (free tier available)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Enhanced financial data
FINANCIAL_DATASETS_API_KEY=your_financial_datasets_api_key_here
```

**Getting API Keys:**
- **OpenRouter**: Sign up at [https://openrouter.ai/](https://openrouter.ai/) for free access to DeepSeek R1
- **Financial Datasets**: Register at [https://financialdatasets.ai/](https://financialdatasets.ai/) for comprehensive financial data

### 3. Run Example

```bash
# Set environment variables
export $(cat .env | xargs)

# Run the investment analysis example
python investment_agent_example.py
```

## 📚 Key Components

### DeepSeek R1 Model Integration

**Model Specifications** (Source: [OpenRouter Documentation](https://openrouter.ai/deepseek/deepseek-r1)):
- **Parameters**: 671B total, 37B active during inference
- **Context Window**: 128,000 tokens
- **Performance**: Comparable to OpenAI's o1 model
- **License**: MIT (free commercial use)
- **Cost**: Free tier available on OpenRouter

```python
from agno.models.openrouter import OpenRouter

model = OpenRouter(
    id="deepseek/deepseek-r1-0528:free",  # Free tier
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,  # Optimal for financial reasoning
    max_tokens=4096
)
```

### Financial Data Sources

**Available Toolkits** (Source: [Agno Documentation](https://docs.agno.com/)):

1. **YFinance Tools** - Free access to Yahoo Finance data
   - Stock prices, historical data, company info
   - Analyst recommendations, financial ratios
   - Technical indicators and news

2. **Financial Datasets API** - Comprehensive financial data
   - Income statements, balance sheets, cash flows
   - SEC filings, institutional ownership
   - Cryptocurrency data, earnings reports

3. **OpenBB Tools** - Open-source financial data platform
   - Market data, company profiles
   - Economic indicators, price targets

### Agent Hierarchy

**Agno Framework Levels** (Source: [Agno Documentation](https://docs.agno.com/introduction)):

- **Level 1**: Data Collection Agents - "Agents with tools and instructions"
- **Level 2**: Analysis Agents - "Agents with knowledge and storage"  
- **Level 3**: Strategy Agents - "Agents with memory and reasoning"
- **Level 4**: Portfolio Teams - "Agent Teams that can reason and collaborate"
- **Level 5**: Execution Workflows - "Agentic Workflows with state and determinism"

## 💼 Usage Examples

### Single Stock Analysis

```python
from investment_agent_example import InvestmentFrameworkDemo

# Initialize framework
demo = InvestmentFrameworkDemo()

# Analyze a single stock
results = await demo.comprehensive_investment_analysis("AAPL")
print(results['final_recommendation'])
```

### Portfolio Optimization

```python
# Define current portfolio
portfolio = {
    "AAPL": 0.20,  # 20% Apple
    "MSFT": 0.15,  # 15% Microsoft
    "GOOGL": 0.10, # 10% Google
    "TSLA": 0.08,  # 8% Tesla
    "NVDA": 0.12   # 12% Nvidia
}

# Get optimization recommendations
results = await demo.orchestrator.comprehensive_investment_analysis(
    "AAPL", 
    portfolio_context=portfolio
)
```

### Custom Agent Creation

```python
from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools

# Create specialized agent
custom_agent = Agent(
    name="ESG Investment Specialist",
    model=model,
    tools=[
        YFinanceTools(company_info=True, company_news=True),
        ReasoningTools(add_instructions=True)
    ],
    instructions=[
        "Focus on ESG (Environmental, Social, Governance) analysis",
        "Evaluate sustainability metrics and social impact",
        "Assess long-term viability and stakeholder alignment"
    ]
)
```

## 🔧 Advanced Features

### Risk Management

- **Value at Risk (VaR)** calculation at 95% and 99% confidence levels
- **Conditional VaR** for tail risk assessment
- **Maximum Drawdown** analysis and stress testing
- **Kelly Criterion** position sizing
- **Correlation analysis** and concentration risk monitoring

### Portfolio Optimization

- **Modern Portfolio Theory** implementation
- **Mean-variance optimization** for efficient frontier
- **Risk budgeting** across positions
- **Rebalancing algorithms** with transaction cost consideration
- **Monte Carlo simulation** for performance projections

### Reasoning and Decision Making

**DeepSeek R1 Reasoning Capabilities** (Source: [DeepSeek R1-0528 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)):
- "Improved reasoning and inference capabilities"
- "Significant performance gains in complex reasoning tasks"
- "Reduced hallucination rate and better function calling support"

The framework leverages these capabilities for:
- Multi-factor investment decision synthesis
- Scenario analysis and probability weighting
- Risk-adjusted return optimization
- Behavioral finance consideration

## 📊 Output Format

The framework provides structured investment recommendations:

```markdown
### Investment Decision: BUY/HOLD/SELL
- **Confidence Level**: 8/10
- **Target Price**: $185.00 (12-month)
- **Stop Loss**: $145.00
- **Position Size**: 5% of portfolio
- **Expected Return**: 15.2% (risk-adjusted)

### Investment Thesis
[Detailed reasoning with supporting analysis]

### Scenario Analysis
- **Base Case** (60%): +12% return
- **Bull Case** (20%): +25% return  
- **Bear Case** (20%): -8% return

### Risk Factors
[Key risks and mitigation strategies]
```

## 🔍 Monitoring and Production

### Performance Tracking

**Agno Monitoring** (Source: [Agno Documentation](https://docs.agno.com/introduction)):
- "Monitor agent sessions and performance in real-time on agno.com"
- Built-in session tracking and debugging
- Performance metrics and usage analytics

### API Deployment

```python
from fastapi import FastAPI
from agno.api import AgentAPI

app = FastAPI(title="AI Investment Agent API")

# Deploy agents as REST APIs
investment_api = AgentAPI(
    agent=orchestrator.orchestrator,
    prefix="/investment-analysis"
)

app.include_router(investment_api.router)
```

### Scalability

**Agno Performance** (Source: [Agno Documentation](https://docs.agno.com/introduction)):
- "Agents instantiate in ~3μs and use ~6.5Kib memory on average"
- "Pre-built FastAPI Routes: 0 to production in minutes"

## 📖 Documentation

- **`ai_investing_agent_framework_guide.md`** - Comprehensive framework documentation
- **`investment_agent_example.py`** - Complete working implementation
- **`requirements.txt`** - Package dependencies
- **`setup.py`** - Automated setup script

## 🔐 Security and Compliance

### Risk Management Controls

- Position size limits (max 10% per position)
- Portfolio concentration limits (max 30% per sector)
- Stop-loss automation (15% default)
- Maximum drawdown monitoring (20% threshold)

### Regulatory Considerations

- Detailed audit trails for all decisions
- Reasoning documentation for compliance
- Risk disclosure and suitability assessment
- Systematic review and backtesting

## 🤝 Contributing

This framework is designed to be extensible. Key areas for enhancement:

1. **Additional Data Sources**: Alternative data, sentiment analysis
2. **Advanced Strategies**: Options strategies, crypto integration
3. **Machine Learning**: Pattern recognition, predictive modeling
4. **Risk Models**: Advanced factor models, stress testing
5. **User Interface**: Web dashboard, mobile integration

## 📄 License

This framework is provided for educational and research purposes. Ensure compliance with:
- Financial regulations in your jurisdiction
- API terms of service for data providers
- Investment advisor regulations if applicable

## ⚠️ Disclaimer

This framework is for educational purposes only and should not be considered as financial advice. Always:
- Conduct your own research
- Consult with qualified financial advisors
- Understand the risks involved in investing
- Test thoroughly before using with real money

## 🆘 Support

For questions and support:
1. Review the comprehensive guide: `ai_investing_agent_framework_guide.md`
2. Check the example implementation: `investment_agent_example.py`
3. Consult Agno documentation: [https://docs.agno.com/](https://docs.agno.com/)
4. OpenRouter documentation: [https://openrouter.ai/docs](https://openrouter.ai/docs)

---

**Built with ❤️ using Agno Framework and DeepSeek R1**
