#!/usr/bin/env python3
"""
Generate markdown report from optimization results.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def load_results(results_dir: Path):
    """Load all result files."""
    results_dir = Path(results_dir)
    
    # Load best params
    best_params_path = results_dir / "best_params.json"
    if not best_params_path.exists():
        raise FileNotFoundError(f"best_params.json not found in {results_dir}")
    
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    # Load trials
    trials_path = results_dir / "trials.jsonl"
    trials = []
    if trials_path.exists():
        with open(trials_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        trials.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Load results CSV
    results_csv_path = results_dir / "results.csv"
    if not results_csv_path.exists():
        raise FileNotFoundError(f"results.csv not found in {results_dir}")
    
    results_df = pd.read_csv(results_csv_path)
    
    return best_params, trials, results_df


def generate_markdown_report(best_params: dict, trials: list, results_df: pd.DataFrame, output_path: Path):
    """Generate markdown report."""
    
    # Calculate statistics
    total_episodes = len(results_df)
    participating = results_df[results_df['weight'] > 0]
    n_trades = len(participating)
    participation_rate = n_trades / total_episodes if total_episodes > 0 else 0
    
    scores = [t.get('score', 0) for t in trials if 'score' in t]
    
    # Start building markdown
    md = []
    md.append("# IPO Trading Strategy - Results Report")
    md.append("")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("---")
    md.append("")
    
    # Best Policy Parameters
    md.append("## 📊 Best Policy Parameters")
    md.append("")
    
    threshold = best_params.get('participate_threshold', 0)
    entry_day = best_params.get('entry_day', 0)
    hold_k = best_params.get('hold_k', 1)
    raw_weight = best_params.get('raw_weight', 0)
    w_max = best_params.get('w_max', 1.0)
    use_vol_cap = best_params.get('use_volume_cap', False)
    vol_cap_mult = best_params.get('vol_cap_mult', 0)
    
    md.append("| Parameter | Value | Description |")
    md.append("|-----------|-------|------------|")
    md.append(f"| `participate_threshold` | {threshold:.6f} | Threshold for participation decision |")
    md.append(f"| `entry_day` | {entry_day} | Entry day (0 = day0, 1 = day1, etc.) |")
    md.append(f"| `hold_k` | {hold_k} | Hold period in days |")
    md.append(f"| `raw_weight` | {raw_weight:.6f} | Base position size |")
    md.append(f"| `w_max` | {w_max:.6f} | Maximum position weight |")
    md.append(f"| `use_volume_cap` | {use_vol_cap} | Whether to use volume-based capping |")
    md.append(f"| `vol_cap_mult` | {vol_cap_mult:.6f} | Volume cap multiplier |")
    md.append("")
    
    # Parameter Interpretation
    md.append("### Participation Strategy")
    md.append("")
    if threshold < 0.1:
        md.append(f"- **Very low threshold** ({threshold:.4f}) → Participates in most IPOs")
        md.append("- Strategy is **aggressive**, taking many opportunities")
    elif threshold < 0.3:
        md.append(f"- **Low threshold** ({threshold:.4f}) → Participates in many IPOs")
        md.append("- Strategy is **moderately selective**")
    elif threshold < 0.7:
        md.append(f"- **Moderate threshold** ({threshold:.4f}) → Selective participation")
        md.append("- Strategy **filters IPOs** based on price movement")
    else:
        md.append(f"- **High threshold** ({threshold:.4f}) → Very selective")
        md.append("- Strategy only participates in **high-confidence IPOs**")
    md.append("")
    
    md.append("### Entry Timing")
    md.append("")
    if entry_day == 0:
        md.append(f"- **Entry on Day 0** (IPO day) → Immediate participation")
        md.append("- Captures **first-day price movements**")
    elif entry_day == 1:
        md.append(f"- **Entry on Day 1** → Waits one day after IPO")
        md.append("- Avoids first-day volatility, enters **after initial pop**")
    else:
        md.append(f"- **Entry on Day {entry_day}** → Delayed entry")
        md.append("- Strategy waits for **market to settle**")
    md.append("")
    
    md.append("### Hold Period")
    md.append("")
    if hold_k == 1:
        md.append(f"- **Hold for {hold_k} day** → Day trading style")
        md.append("- **Quick in-and-out**, captures short-term moves")
    elif hold_k <= 3:
        md.append(f"- **Hold for {hold_k} days** → Short-term holding")
        md.append("- Captures **multi-day momentum**")
    elif hold_k <= 7:
        md.append(f"- **Hold for {hold_k} days** → Medium-term holding")
        md.append("- Allows time for **price discovery**")
    else:
        md.append(f"- **Hold for {hold_k} days** → Long-term holding")
        md.append("- **Patient strategy**, waits for longer trends")
    md.append("")
    
    md.append("### Position Sizing")
    md.append("")
    if raw_weight < 0.1:
        md.append(f"- **Base weight:** {raw_weight:.4f} ({raw_weight*100:.2f}% of portfolio)")
        md.append("- **Conservative sizing**, small positions")
    elif raw_weight < 0.3:
        md.append(f"- **Base weight:** {raw_weight:.4f} ({raw_weight*100:.2f}% of portfolio)")
        md.append("- **Moderate sizing**, balanced risk")
    else:
        md.append(f"- **Base weight:** {raw_weight:.4f} ({raw_weight*100:.2f}% of portfolio)")
        md.append("- **Aggressive sizing**, larger positions")
    md.append(f"- **Maximum weight cap:** {w_max:.4f} ({w_max*100:.2f}%)")
    if use_vol_cap:
        md.append(f"- **Volume cap enabled:** {vol_cap_mult:.4f}x daily volume")
        md.append("- Limits position size based on liquidity")
    else:
        md.append("- **Volume cap disabled** → No liquidity constraints")
    md.append("")
    md.append("---")
    md.append("")
    
    # Performance Analysis
    md.append("## 📈 Performance Analysis")
    md.append("")
    
    md.append("### Trade Statistics")
    md.append("")
    md.append(f"- **Total IPOs analyzed:** {total_episodes}")
    md.append(f"- **IPOs traded:** {n_trades} ({participation_rate*100:.1f}%)")
    md.append(f"- **IPOs skipped:** {total_episodes - n_trades} ({(1-participation_rate)*100:.1f}%)")
    md.append("")
    
    if n_trades > 0:
        gross_ret = participating['gross_ret']
        net_ret = participating['net_ret']
        costs = participating['cost']
        
        md.append("### Return Statistics")
        md.append("")
        md.append(f"- **Average gross return:** {gross_ret.mean():.4f} ({gross_ret.mean()*100:.2f}%)")
        md.append(f"- **Average net return:** {net_ret.mean():.4f} ({net_ret.mean()*100:.2f}%)")
        md.append(f"- **Total gross return:** {gross_ret.sum():.4f} ({gross_ret.sum()*100:.2f}%)")
        md.append(f"- **Total net return:** {net_ret.sum():.4f} ({net_ret.sum()*100:.2f}%)")
        md.append("")
        md.append(f"- **Best trade:** {gross_ret.max():.4f} ({gross_ret.max()*100:.2f}%)")
        md.append(f"- **Worst trade:** {gross_ret.min():.4f} ({gross_ret.min()*100:.2f}%)")
        md.append(f"- **Median return:** {gross_ret.median():.4f} ({gross_ret.median()*100:.2f}%)")
        md.append("")
        
        win_rate = (gross_ret > 0).sum() / n_trades
        md.append(f"- **Win rate:** {win_rate*100:.1f}% ({int(win_rate * n_trades)} wins, {int((1-win_rate) * n_trades)} losses)")
        md.append("")
        
        md.append("### Cost Analysis")
        md.append("")
        md.append(f"- **Total costs:** {costs.sum():.4f} ({costs.sum()*100:.2f}%)")
        md.append(f"- **Average cost per trade:** {costs.mean():.4f} ({costs.mean()*100:.2f}%)")
        cost_impact = (costs.sum() / gross_ret.sum() * 100) if gross_ret.sum() != 0 else 0
        md.append(f"- **Cost impact:** {cost_impact:.1f}% of gross returns")
        md.append("")
        
        md.append("### Risk Metrics")
        md.append("")
        if len(net_ret) > 0:
            std_dev = net_ret.std()
            sharpe = (net_ret.mean() / std_dev) if std_dev > 0 else 0
            md.append(f"- **Volatility (std dev):** {std_dev:.4f} ({std_dev*100:.2f}%)")
            md.append(f"- **Sharpe ratio (approx):** {sharpe:.2f}")
        md.append("")
    else:
        md.append("⚠️ **No trades executed** - strategy chose to skip all IPOs")
        md.append("")
    
    md.append("---")
    md.append("")
    
    # Optimization Trials Analysis
    if trials:
        md.append("## 🔍 Optimization Trials Analysis")
        md.append("")
        
        if scores:
            md.append("### Trial Statistics")
            md.append("")
            md.append(f"- **Total trials:** {len(trials)}")
            md.append(f"- **Best score:** {max(scores):.6f}")
            md.append(f"- **Worst score:** {min(scores):.6f}")
            md.append(f"- **Average score:** {np.mean(scores):.6f}")
            md.append(f"- **Median score:** {np.median(scores):.6f}")
            md.append(f"- **Score std dev:** {np.std(scores):.6f}")
            md.append("")
            
            # Top 3 trials
            sorted_trials = sorted(trials, key=lambda x: x.get('score', -float('inf')), reverse=True)
            md.append("### Top 3 Trials")
            md.append("")
            for i, trial in enumerate(sorted_trials[:3], 1):
                score = trial.get('score', 0)
                metrics = trial.get('metrics', {})
                md.append(f"#### Trial #{i}: Score = {score:.6f}")
                md.append("")
                if metrics:
                    md.append(f"- E[R]: {metrics.get('E[R]', 0):.6f}")
                    md.append(f"- CVaR: {metrics.get('CVaR', 0):.6f}")
                    md.append(f"- E[Cost]: {metrics.get('E[Cost]', 0):.6f}")
                    md.append(f"- MDD: {metrics.get('MDD', 0):.6f}")
                md.append("")
            
            # Parameter ranges
            md.append("### Parameter Exploration Ranges")
            md.append("")
            thresholds = [t.get('params', {}).get('participate_threshold', 0) for t in trials]
            hold_ks = [t.get('params', {}).get('hold_k', 1) for t in trials]
            weights = [t.get('params', {}).get('raw_weight', 0) for t in trials]
            
            md.append(f"- `participate_threshold`: [{min(thresholds):.4f}, {max(thresholds):.4f}]")
            md.append(f"- `hold_k`: [{min(hold_ks)}, {max(hold_ks)}]")
            md.append(f"- `raw_weight`: [{min(weights):.4f}, {max(weights):.4f}]")
            md.append("")
        
        md.append("---")
        md.append("")
    
    # Key Insights
    md.append("## 💡 Key Insights & Interpretation")
    md.append("")
    
    md.append("### Strategy Profile")
    md.append("")
    if participation_rate < 0.1:
        md.append("- **Very conservative strategy** - rarely participates")
        md.append("- Low risk, but may miss opportunities")
    elif participation_rate < 0.3:
        md.append("- **Selective strategy** - participates in ~1/4 to 1/3 of IPOs")
        md.append("- Balanced approach between risk and opportunity")
    elif participation_rate < 0.7:
        md.append("- **Active strategy** - participates in majority of IPOs")
        md.append("- Higher exposure, captures more opportunities")
    else:
        md.append("- **Very active strategy** - participates in most IPOs")
        md.append("- Maximum exposure, highest risk/reward")
    md.append("")
    
    if n_trades > 0:
        avg_return = participating['net_ret'].mean()
        win_rate = (participating['gross_ret'] > 0).sum() / n_trades
        
        md.append("### Performance Assessment")
        md.append("")
        if avg_return > 0.01:
            md.append("- ✅ **Strong positive returns** → Strategy is profitable")
        elif avg_return > 0:
            md.append("- ✅ **Modest positive returns** → Strategy shows promise")
        elif avg_return > -0.01:
            md.append("- ⚠️ **Near break-even** → Strategy needs refinement")
        else:
            md.append("- ❌ **Negative returns** → Strategy needs significant improvement")
        md.append("")
        
        if win_rate > 0.6:
            md.append(f"- ✅ **High win rate** ({win_rate*100:.1f}%) → Good trade selection")
        elif win_rate > 0.5:
            md.append(f"- ⚠️ **Moderate win rate** ({win_rate*100:.1f}%) → Balanced outcomes")
        else:
            md.append(f"- ❌ **Low win rate** ({win_rate*100:.1f}%) → Need better entry criteria")
        md.append("")
    
    md.append("### Recommendations")
    md.append("")
    if entry_day == 0 and hold_k == 1:
        md.append("- Current strategy: **Day-0 entry, 1-day hold**")
        md.append("- 💡 Consider: Testing longer holds (3-5 days) for better trend capture")
    elif hold_k <= 2:
        md.append("- Current strategy: **Short-term holding**")
        md.append("- 💡 Consider: Testing if extending hold period improves returns")
    else:
        md.append("- Current strategy: **Medium-to-long-term holding**")
        md.append("- 💡 Consider: Testing shorter holds to reduce exposure time")
    md.append("")
    
    if threshold < 0.1:
        md.append("- Very low participation threshold")
        md.append("- 💡 Consider: Increasing threshold to be more selective")
    elif threshold > 0.5:
        md.append("- High participation threshold")
        md.append("- 💡 Consider: Lowering threshold to capture more opportunities")
    md.append("")
    
    md.append("---")
    md.append("")
    md.append("## 📝 Summary")
    md.append("")
    md.append(f"- **Strategy Type:** {'Conservative' if participation_rate < 0.3 else 'Active' if participation_rate > 0.7 else 'Selective'}")
    md.append(f"- **Entry:** Day {entry_day}, Hold: {hold_k} days")
    md.append(f"- **Participation Rate:** {participation_rate*100:.1f}%")
    if n_trades > 0:
        md.append(f"- **Average Return:** {participating['net_ret'].mean():.4f} ({participating['net_ret'].mean()*100:.2f}%)")
        md.append(f"- **Win Rate:** {(participating['gross_ret'] > 0).sum() / n_trades * 100:.1f}%")
    if scores:
        md.append(f"- **Best Score:** {max(scores):.6f}")
    md.append("")
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))
    
    print(f"✅ Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report from optimization results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing results files (default: results/)")
    parser.add_argument("--output", type=str, default="results/report.md",
                        help="Output markdown file path (default: results/report.md)")
    
    args = parser.parse_args()
    
    try:
        best_params, trials, results_df = load_results(args.results_dir)
        generate_markdown_report(best_params, trials, results_df, args.output)
        return 0
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you've run the optimization first:")
        print("  python3 run_week3.py --data synth --trials 20")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
