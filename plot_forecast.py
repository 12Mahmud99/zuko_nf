import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_forecast(
    forecast_file,
    window_idx=0,
    series_idx=0,
    show_samples=True,
    alpha=0.3,
    output=None
):
    data = np.load(forecast_file)
    full_gt = data['full_ground_truth']     # (n_windows, n_series, context_len+pred_len)
    forecasts = data['forecasts']           # (n_windows, n_samples, n_series, pred_len)
    start_indices = data['start_indices']

    if window_idx >= len(full_gt):
        raise ValueError(f"window_idx {window_idx} out of range")
    if series_idx >= full_gt.shape[1]:
        raise ValueError(f"series_idx {series_idx} out of range")

    # Full ground truth for this window/series (context + future)
    full_series = full_gt[window_idx, series_idx, :]   # (context_len+pred_len)
    pred_len = forecasts.shape[-1]
    context_len = len(full_series) - pred_len

    context = full_series[:context_len]
    future_gt = full_series[context_len:]   # ground truth for forecast period

    # Forecast samples for this window and series
    fc_samples = forecasts[window_idx, :, series_idx, :]  # (n_samples, pred_len)
    mean_forecast = fc_samples.mean(axis=0)
    std_forecast = fc_samples.std(axis=0)

    # Time indices
    start = start_indices[window_idx]
    context_time = np.arange(start, start + context_len)
    forecast_time = np.arange(start + context_len, start + context_len + pred_len)
    combined_time = np.concatenate([context_time, forecast_time])

    plt.figure(figsize=(14, 6))

    # Plot the entire ground truth as a single black line (context + actual future)
    combined_gt = np.concatenate([context, future_gt])
    plt.plot(combined_time, combined_gt, 'black', linewidth=2, label='Actual future')

    # Forecast samples
    if show_samples:
        for i in range(min(fc_samples.shape[0], 50)):
            plt.plot(forecast_time, fc_samples[i], 'r-', alpha=alpha, linewidth=0.8)

    # Mean forecast and uncertainty
    plt.plot(forecast_time, mean_forecast, 'r-', linewidth=2, label='Mean forecast')
    plt.fill_between(forecast_time, mean_forecast - std_forecast,
                     mean_forecast + std_forecast, color='r', alpha=0.2, label='±1 std')

    plt.title(f"Forecast for window {window_idx}, series {series_idx} (start index = {start})")
    plt.xlabel("Time step")
    plt.ylabel("Value (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_file", "-f", required=True)
    parser.add_argument("--window", "-w", type=int, default=0)
    parser.add_argument("--series", "-s", type=int, default=0)
    parser.add_argument("--no_samples", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    plot_forecast(
        forecast_file=args.forecast_file,
        window_idx=args.window,
        series_idx=args.series,
        show_samples=not args.no_samples,
        alpha=args.alpha,
        output=args.output
    )