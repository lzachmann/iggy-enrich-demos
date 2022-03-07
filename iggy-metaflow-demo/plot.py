import random
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional
import plotly.graph_objects as go
import plotly
from sklearn.neural_network import MLPRegressor


def regress_obs_vs_pred(
    model: MLPRegressor,
    X_test: np.array,
    y_test: np.array,
    file_path: str,
    auto_open: bool = False,
    scaled_mean: Optional[float] = None,
    scaled_std: Optional[float] = None,
) -> Dict:

    """Create observed vs. predicted plot"""

    y_hat = model.predict(X_test)
    dirname = os.path.dirname(file_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    if len(y_test) > 1_000:
        idx = random.sample(range(len(y_test)), 1_000)
    else:
        idx = list(range(len(y_test)))
    x_raw = y_hat.iloc[idx] if isinstance(y_hat, pd.Series) else y_hat[idx]
    y_raw = y_test.iloc[idx] if isinstance(y_test, pd.Series) else y_test[idx]

    # Conditionally unscale and untransform the y's
    if scaled_mean:
        x_unscaled = x_raw * scaled_std + scaled_mean
        y_unscaled = y_raw * scaled_std + scaled_mean
        x = 10**x_unscaled
        y = 10**y_unscaled
    else:
        x = x_raw
        y = y_raw

    plot_range = [np.min((x, y)), np.max((x, y))]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
        ),
    )
    fig.add_shape(
        type="line",
        x0=plot_range[0], y0=plot_range[0], x1=plot_range[1], y1=plot_range[1],
        line=dict(
            color="Black",
            width=2,
        )
    )
    fig.update_layout(
        xaxis_title='Predicted',
        yaxis_title='Observed',
        template='none',
    )
    fig.write_image(file_path)
    plotly.offline.plot(fig, filename=file_path, auto_open=auto_open)

    return fig
