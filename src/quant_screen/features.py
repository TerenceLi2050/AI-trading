from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market import trade_flags
from .pit import FUNDAMENTAL_FACTORS


# Signs express the economic hypothesis behind the transparent baseline.
PRICE_FACTORS = {
    "momentum_120_20": 1, "momentum_60_5": 1, "residual_momentum_60": 1,
    "trend": 1, "frog_in_pan": 1, "scaled_momentum": 1, "reversal_5": 1,
    "low_volatility": 1, "drawdown_60": 1, "low_downside_risk": 1,
    "low_amihud": 1, "liquidity": 1, "steady_turnover": 1, "turnover_shock": -1,
    "overnight_return_20": -1, "intraday_return_20": 1, "close_location_20": 1,
}
BASELINE_GROUPS = [
    ["momentum_120_20", "momentum_60_5", "residual_momentum_60", "trend", "frog_in_pan", "scaled_momentum"],
    ["low_volatility", "drawdown_60", "low_downside_risk"],
    ["low_amihud", "liquidity", "steady_turnover", "turnover_shock"],
    ["reversal_5", "overnight_return_20", "intraday_return_20", "close_location_20"],
]
RANK_PREFIX = "rank_"


@dataclass
class Features:
    panel: pd.DataFrame
    columns: list[str]
    market: pd.DataFrame


def build_features(quotes: pd.DataFrame, calendar: pd.DataFrame, member: pd.Series, industry: pd.Series,
                   coverage: pd.DataFrame, financial: pd.DataFrame | None, config: dict) -> Features:
    flags = trade_flags(quotes)
    sessions = calendar.index
    codes = pd.Index(quotes["code"].unique()).sort_values()
    row = sessions.get_indexer(quotes["date"])
    col = codes.get_indexer(quotes["code"])

    def matrix(column: str, mask: pd.Series = flags["bar_ok"]) -> pd.DataFrame:
        values = np.full((len(sessions), len(codes)), np.nan)
        values[row, col] = quotes[column].where(mask).to_numpy(dtype=float, na_value=np.nan)
        return pd.DataFrame(values, index=sessions, columns=codes)

    close = matrix("adj_close")
    opened = matrix("adj_open")
    high = matrix("adj_high")
    low = matrix("adj_low")
    amount = matrix("amount")
    turnover = matrix("turnover").where(lambda frame: frame.ge(0))
    returns = close.div(close.shift()).sub(1)
    market_ret = calendar["market_close"].pct_change(fill_method=None)
    beta = returns.rolling(60, min_periods=45).cov(market_ret).div(market_ret.rolling(60, min_periods=45).var(), axis=0)
    residual = returns - beta.shift(1).mul(market_ret, axis=0)
    vol = returns.rolling(60, min_periods=48).std()
    adv = amount.rolling(20, min_periods=15).median()
    range_price = (high - low).where(lambda frame: frame.gt(0))
    mom = close.shift(5).div(close.shift(60)).sub(1)
    signs = np.sign(returns)
    raw = {
        "momentum_120_20": close.shift(20).div(close.shift(120)).sub(1),
        "momentum_60_5": mom,
        "residual_momentum_60": residual.rolling(60, min_periods=45).mean(),
        "trend": close.rolling(20, min_periods=16).mean().div(close.rolling(120, min_periods=100).mean()).sub(1),
        "frog_in_pan": mom.abs() * signs.shift(5).rolling(55, min_periods=44).mean(),
        "scaled_momentum": mom.div(vol.shift(5).where(lambda frame: frame.gt(0))),
        "reversal_5": close.shift(5).div(close).sub(1),
        "low_volatility": -vol,
        "drawdown_60": close.div(close.rolling(60, min_periods=48).max()).sub(1),
        "low_downside_risk": -returns.clip(upper=0).pow(2).rolling(60, min_periods=48).mean().pow(0.5),
        "low_amihud": -returns.abs().div(amount).rolling(20, min_periods=15).mean(),
        "liquidity": np.log(adv),
        "steady_turnover": -turnover.rolling(20, min_periods=15).std().div(turnover.rolling(20, min_periods=15).mean()),
        "turnover_shock": turnover.rolling(5, min_periods=4).mean().div(turnover.rolling(60, min_periods=48).mean()),
        "overnight_return_20": opened.div(close.shift()).sub(1).rolling(20, min_periods=15).mean(),
        "intraday_return_20": close.div(opened).sub(1).rolling(20, min_periods=15).mean(),
        "close_location_20": (close - low).div(range_price).rolling(20, min_periods=15).mean(),
    }

    def values(frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy()[row, col]

    settings = config["eligibility"]
    panel = pd.DataFrame({name: values(frame) for name, frame in raw.items()}, index=quotes.index)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel[["date", "code", "raw_close", "is_st", "price_source", "snapshot_industry"]] = quotes[
        ["date", "code", "raw_close", "is_st", "price_source", "snapshot_industry"]]
    panel["industry"] = industry
    panel["member"] = member
    panel["bar_ok"] = flags["bar_ok"]
    panel["adv20"] = values(adv)
    panel["vol60"] = values(vol)
    panel["valid_bars_60"] = values(close.rolling(60, min_periods=1).count())
    panel["cumulative_bars"] = values(close.notna().cumsum())
    panel["recent_extreme_move"] = values(returns.abs().rolling(20, min_periods=15).max())
    panel["close_to_high"] = values(close.div(close.rolling(252, min_periods=120).max()).sub(1))
    panel["tradable_history"] = (panel["valid_bars_60"].ge(settings["min_valid_bars_60"])
                                 & panel["cumulative_bars"].ge(settings["min_listing_sessions"]))
    good = (member & flags["bar_ok"] & ~panel["is_st"].eq(1) & panel["tradable_history"]
            & panel["adv20"].ge(settings["min_median_amount"]) & panel["raw_close"].ge(settings["min_raw_price"])
            & panel["recent_extreme_move"].le(settings["max_jump"]))
    covered = (coverage["expected"].isna() | (coverage["coverage"].ge(config["data"]["minimum_member_coverage"])
               & coverage["expected"].ge(config["data"]["minimum_member_count"])))
    good &= panel["date"].map(covered)
    panel["eligible"] = good & good.groupby(panel["date"]).transform("sum").ge(config["data"]["minimum_cross_section"])
    names = [f"{RANK_PREFIX}{name}" for name in PRICE_FACTORS]
    # Percentile ranks are computed independently after the same-day eligibility check.
    ranks = panel.loc[:, list(PRICE_FACTORS)].where(panel["eligible"], axis=0).groupby(panel["date"]).rank(pct=True)
    panel[names] = ranks.astype("float32")
    group_scores = []
    for group in BASELINE_GROUPS:
        oriented = [panel[f"{RANK_PREFIX}{name}"] if PRICE_FACTORS[name] > 0 else 1 - panel[f"{RANK_PREFIX}{name}"]
                    for name in group]
        group_scores.append(pd.concat(oriented, axis=1).mean(axis=1))
    panel["baseline"] = pd.concat(group_scores, axis=1).mean(axis=1)

    if financial is None:
        panel["fundamentals_ok"] = panel["negative_equity"] = panel["ttm_loss"] = False
        panel["quality_score"] = np.nan
    else:
        panel = pd.concat([panel, financial], axis=1)
        peer_mask = panel["eligible"] & panel["fundamentals_ok"]
        profitability = ["roe_ttm", "roa_ttm", "gross_profit_assets", "cash_return_assets", "low_accruals"]
        values = panel[FUNDAMENTAL_FACTORS].where(peer_mask, axis=0)
        peer_count = peer_mask.groupby([panel["date"], panel["industry"]]).transform("sum")
        f_ranks = values.groupby([panel["date"], panel["industry"]]).rank(pct=True)
        f_ranks = f_ranks.where(peer_count.ge(8), values.groupby(panel["date"]).rank(pct=True), axis=0)
        financial_company = panel["industry"].str.contains("银行|证券|金融|保险|多元金融", na=False)
        cash_conversion = panel["cfo_ttm"].div(panel["net_income_ttm"].where(panel["net_income_ttm"].gt(0)))
        operating_score = f_ranks[profitability].mean(axis=1).where(
            f_ranks[profitability].count(axis=1).ge(3) & cash_conversion.ge(settings["minimum_cash_conversion"]))
        finance_score = f_ranks[["roe_ttm", "roa_ttm", "revenue_growth"]].mean(axis=1)
        panel["quality_score"] = operating_score.where(~financial_company, finance_score).where(panel["net_income_ttm"].gt(0))

    market = calendar.copy()
    market["return_20"] = calendar["market_close"].pct_change(20)
    market["annualized_vol_60"] = market_ret.rolling(60).std() * np.sqrt(252)
    market["above_ma120"] = calendar["market_close"].gt(calendar["market_close"].rolling(120).mean())
    market["eligible_count"] = panel["eligible"].groupby(panel["date"]).sum().reindex(sessions, fill_value=0)
    market["positive_momentum_breadth"] = panel["momentum_60_5"].gt(0).where(panel["eligible"]).groupby(panel["date"]).mean()
    return Features(panel, names, market)
