#!/usr/bin/env python3

from dash import dcc
from dash import html

from app import get_file

selections = get_file('sim_names')
slow_pairs = get_file('slow_pairs')

page_1_layout = html.Div(children=[
    html.H1(children="MD trajectory analysis"),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="dd-selections",
                options=[
                    {"label": selection, "value": selection} for selection in selections if 'PDB' not in selection
                ],
                value=[selection for selection in selections if 'PDB' not in selection][0],
                persistence=True,
                clearable=False
            ),

            dcc.Dropdown(
                id="dd-local",
                options=[
                    {"label": resid, "value": resid} for resid in slow_pairs
                ],
                value=slow_pairs[0],
                persistence=True,
                clearable=True
            ),

            dcc.RangeSlider(
                id="slider-time",
                min=0,
                max=50000,
                step=1000,
                marks={
                    1: "0.1ns",
                    5000: "500ns",
                    10000: "1us",
                    20000: "2us",
                    30000: "3us",
                    40000: "4us",
                    50000: "5us",
                },
                value=[0, 50000],
                persistence=True
            ),
        ], style={"width": "45%", "display": "inline-block"}),

        dcc.Store(id='data-tics'),
        dcc.Store(id='data-metastable_traj'),
        dcc.Store(id='data-its'),

        html.Div([
            dcc.Graph(
                id="plot-its",
                className="six columns"),

            dcc.Graph(
                id="plot-local",
                className="six columns"),
        ]),

        dcc.RadioItems(
            id="radio-traj_length",
            options=[
                {"label": "full trajectory length (slower loading)", "value": "full"},
                {"label": "reduced trajectory length", "value": "reduced"}
            ],
            value="reduced",
            persistence=True,
        )
    ], style={"width": "100%", "display": "inline-block"}
    ),
    html.Div(id='home-content'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/home-content')
])