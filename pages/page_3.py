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
            )]),

        dcc.Store(id='data-tcf_full'),
        dcc.Store(id='data-tcf_1'),
        dcc.Store(id='data-tcf_2'),

        html.Div([
            dcc.Graph(
                id="plot-tcf",
                className="six columns"),

            # dcc.Graph(
            #     id="plot-local",
            #     className="six columns"),
        ]),
    ], style={"width": "100%", "display": "inline-block"}
    ),
    html.Div(id='home-content'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/home-content')
])