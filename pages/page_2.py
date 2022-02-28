#!/usr/bin/env python3

from dash import dcc
from dash import html

from app import get_file

sim_names = get_file('sim_names')
resids = get_file('resids')

page_2_layout = html.Div(children=[
        html.Div([
            html.H1(children='xpk xplorer'),

            html.Div(children='''
                            Select the desired cross-peak:
                            '''),
            html.Div([
                dcc.Dropdown(
                    id='dd-ffs',
                    options=[
                        {"label": ff, "value": ff} for ff in sim_names
                    ],
                    value=sim_names[0],
                    persistence=True,
                    clearable=False,
                    multi=True
                ),

                html.Div([
                html.Div(
                    dcc.Dropdown(
                        id='dd-resid1',
                        options=[
                            {"label": resid, "value": str(resid)} for idx, resid in enumerate(resids)
                        ],
                        value=str(resids[0]),
                        persistence=True,
                        clearable=False
                    ), style={'width': '50%', 'display': 'inline-block'}),

                html.Div(
                    dcc.Dropdown(
                        id='dd-resid2',
                        options=[
                            {"label": resid, "value": str(resid)} for idx, resid in enumerate(resids)
                        ],
                        value=str(resids[0]),
                        persistence=True,
                        clearable=False
                    ), style={'width': '50%', 'display': 'inline-block'})
                ]),

                dcc.RadioItems(
                    id="radio-averaging",
                    options=[
                        {"label": "slow only", "value": "slow"},
                        {"label": "slow and fast", "value": "slow_fast"},
                    ],
                    value="slow_fast"),
            html.Hr(),

            dcc.Graph(
                    id='plot-build-up',
                ),
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),
    html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 3', href='/page-3'),
    html.Br(),
    dcc.Link('Go back to home', href='/home-content')
])