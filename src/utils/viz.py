# viz.py
import plotly.graph_objects as go
import plotly.colors as pc
from PIL import Image
import numpy as np
import cv2

def overlay_mask(img, mask, alpha=0.3):
    out = img.copy()
    m = mask.astype(bool)
    red = np.array([255,0,0], np.uint8)
    out[m] = ((1-alpha)*out[m] + alpha*red).astype(np.uint8)
    cnts,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255,255,255), 1)
    return out

def plot_detections(image, boxes, preds, uirevision="stable", use_invisible_markers=True):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=Image.fromarray(image),
        x=0, y=0,
        sizex=image.shape[1], sizey=image.shape[0],
        xref="x", yref="y",
        layer="below",
        sizing="stretch"
    ))

    fig.update_xaxes(visible=False, range=[0, image.shape[1]], scaleanchor='y', scaleratio=1)
    fig.update_yaxes(visible=False, range=[image.shape[0], 0])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        dragmode="select",
        uirevision=uirevision,
        modebar_add=["drawrect", "select", "eraseshape"],
        hovermode="closest"  # default, explicit for clarity
    )

    colors = pc.qualitative.Plotly
    for i, (x, y, w, h) in enumerate(boxes):
        xs = [x, x + w, x + w, x, x]
        ys = [y, y, y + h, y + h, y]

        hover_html = (
            f"ROI #{i}<br>"
            f"{preds[i]}<br>"
            f"({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})"
        )

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers" if use_invisible_markers else "lines",
            marker=dict(size=6, opacity=0) if use_invisible_markers else None,
            line=dict(color=colors[i % len(colors)], width=2),
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            hoveron="fills+points",          # enable hovering over the filled polygon (and markers if present)
            hovertemplate=hover_html + "<extra></extra>",  # suppress trace name box
            name=""                          # keep empty so no stray label
        ))

    return fig


