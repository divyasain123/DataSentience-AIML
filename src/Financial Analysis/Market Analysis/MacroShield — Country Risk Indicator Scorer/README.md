# 🔍 MacroShield — Country Risk Indicator Scorer

MacroShield classifies countries into **Low**, **Medium**, or **High** macroeconomic risk based on daily financial indicators.

## 🎯 Target
Daily country-level risk scoring using:
- Exchange Rate Volatility
- Equity Volatility
- Bond Yield Levels
- Oil Price
- Yield Spread Change

## 📊 Model
- **Random Forest Classifier**
- Custom rules used to generate `Risk_Level` label
- Features: 5-day rolling volatilities and macro indicators

## 🚀 How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
