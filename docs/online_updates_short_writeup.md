## Online Policy: Why It Underperformed and What We’ll Change

Our naive online updates underperformed for three main reasons. First, the trigger signal (`val_loss` improvement) was weakly related to future returns; the plot `figures/online_evaluation/online_update_val_delta_vs_future_return.png` shows this directly, so many updates were likely noise. Second, we optimized short-horizon validation behavior, but the true target is forward net performance (return, drawdown, and cost-adjusted utility). Third, fixed cadence updates can be mistimed: weekly/daily retraining forces model changes even when regimes are stable, which increases drift and overfitting risk.

Train-window context: each sample uses `window_len=126` days. In rolling mode (`online_train_lookback=252`, `val_frac=0.2`), each online step uses about 252 windows total (~201 train, ~51 validation), which is relatively small for frequent updates in a noisy, nonstationary setting.

To improve performance, we are moving to confidence-gated updates (`IPO_UPDATE_GATE_MODE=confidence`) and only applying updates when improvement passes thresholds.

A/B test plan: compare **cadence-only** vs **confidence-gated** with all other settings fixed. We will select the winner by net Sharpe, net total return, max drawdown, and update efficiency (`difference_update_minus_no_update`, accept rate, and number of applied updates).
