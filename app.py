import argparse
from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import flask
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.graph_objs import Scatter, Figure

from utils import sqlite_utils as sql

ROOTDIR = Path(__file__).parent


def get_sim_names():
    _table_names = sql.fetch_table_names(db)
    _table_names = list(set([table.split('__')[0] for table in _table_names]))
    return [sim for sim in _table_names]


def get_file_dir(filename):
    return Path(ROOTDIR, "data", filename).with_suffix('.pkl')


def save_file(filename, container):
    filedir = Path(ROOTDIR, 'data', filename).with_suffix('.pkl')
    file = open(filedir, "wb")
    pickle.dump(container, file)


def get_file(filename):
    file = open(get_file_dir(filename), 'rb')
    return pickle.load(file)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


### dependent dropdown menu
@app.callback(
    Output('dd-local', 'options'),
    Input('dd-selections', 'value'))
def set_pair_options(selection):
    local_names = sql.fetch_column_names(db, f"{selection}__pair__PC1")
    return [{'label': i, 'value': i} for i in local_names]


@app.callback(
    Output('dd-local', 'value'),
    Input('dd-local', 'options'))
def set_pair_value(available_options):
    return available_options[0]['value']

###

# Page 2 callback
@app.callback( [Output('data-tics', 'data'),
                Output('data-metastable_traj', 'data'),
                Output('data-its', 'data'),
                ],
               [Input("dd-selections", "value"),
                Input("dd-local", "value"),
                Input("slider-time", "value"),
                Input("radio-traj_length", "value"),
                ])
def get_data(selection, methyl_pair, time, traj_length):
    connection = sqlite3.connect(db)

    PC1s_local = pd.read_sql_query(f"""SELECT {methyl_pair} FROM {selection}__pair__PC1""", connection)
    PC2s_local = pd.read_sql_query(f"""SELECT {methyl_pair} FROM {selection}__pair__PC2""", connection)

    its = pd.read_sql_query(f"""SELECT {methyl_pair} FROM {selection}__its""", connection)
    its = its[methyl_pair]
    its = its.to_numpy()

    metastable_traj = pd.read_sql_query(f"""SELECT {methyl_pair} FROM {selection}__labels""", connection)
    metastable_traj = metastable_traj[methyl_pair]
    metastable_traj = metastable_traj.to_numpy()

    connection.close()

    df_full_time = pd.concat([PC1s_local, PC2s_local],
                             axis=1,
                             keys=["PC1", "PC2"]).swaplevel(0, 1,
                                                            axis=1).sort_index(axis=1)

    start, stop = time
    tics = df_full_time[(df_full_time.index >= start) & (df_full_time.index <= stop)]
    metastable_traj = metastable_traj[start:stop]

    if traj_length == "reduced":
        tics = tics.iloc[::10, :]

        metastable_traj = metastable_traj[::10]

    tics = tics[methyl_pair].to_numpy()
    return [tics, metastable_traj, its]


@app.callback(
    dash.dependencies.Output("plot-local", "figure"),
    [Input("data-tics", "data"),
     Input("data-metastable_traj", "data"),
     ])
def plot_metastable_assignments(tics, metastable_traj):
    fig = Figure()

    n_states = max(metastable_traj) + 1
    tics = np.array(tics)
    metastable_traj = np.array(metastable_traj)

    for i in range(n_states):
        idx = np.argwhere(metastable_traj == i).T[0]
        tics_state = tics[idx]

        x = tics_state[:, 0]
        y = tics_state[:, 1]

        fig.add_trace(Scatter(
            x=x,
            y=y,
            xaxis="x",
            yaxis="y",
            name=f'State {i + 1}',
            mode="markers",
            marker=dict(
                size=4,
                color=ms_colors[i]
            ),
            text=f'State {i + 1}'
        ))

    fig.update_layout(
        xaxis=dict(
            title="tIC1"),
        yaxis=dict(
            title="tIC2"),
    )

    return fig


@app.callback(
    dash.dependencies.Output("plot-its", "figure"),
    [Input("data-its", "data"),])
def plot_its(its):
    fig = Figure()
    lag = list(range(1, 11))

    timescales = np.array(its) * 0.1  # dt = 0.1 [ns]

    # TAUC traces
    tauc = 37  # ns
    for idx, scale in enumerate([0.1, 1, 10]):
        fig.add_trace(Scatter(
            x=lag,
            y=[scale * tauc for _ in range(len(lag))],
            xaxis="x",
            yaxis="y",
            mode="lines",
            line=dict(
                width=3,
                color=f"rgba(91, 44, 111, {(idx + 1) * 0.33})",
                dash="dot"
            ),
            name=f"≈ {scale} x tauc"
        ))

    # LAG == LAG
    fig.add_trace(Scatter(
        x=lag,
        y=lag,
        xaxis="x",
        yaxis="y",
        fill="tozeroy", fillcolor="grey",
        mode="lines",
        line=dict(
            width=4,
            color="black"
        ),
        name="Lag time border"
    ))

    fig.add_trace(Scatter(
        x=lag,
        y=pd.Series(timescales),
        xaxis="x",
        yaxis="y",
        name=f"Process 1",
        mode="lines",
        line=dict(
            width=4,
            color=ms_colors[3]
        ),
    ))

    fig.update_yaxes(type="log")

    fig.update_layout(
        xaxis=dict(
            title="Lag time [ns]"),
        yaxis=dict(
            title="Implied Timescale [ns]"),
    )

    return fig


# Page 2
@app.callback(
    dash.dependencies.Output('plot-build-up', 'figure'),
    [dash.dependencies.Input('dd-ffs', 'value'),
     dash.dependencies.Input('dd-resid1', 'value'),
     dash.dependencies.Input('dd-resid2', 'value'),
     dash.dependencies.Input('radio-averaging', 'value'),])
def make_xpk_plot(ff_names, resid1, resid2, averaging):

    if type(ff_names) == str:
        ff_names = [ff_names]

    fig = Figure()

    resid1, resid2 = sorted([int(resid1), int(resid2)])

    for ff_name in ff_names:
        # get the data
        data = sql.fetch_table_where2(db, f'{ff_name}__calcnoesy', 'resid1', resid1, 'resid2',
                                                     resid2).tms_intsts
        data = data.to_numpy()[0]
        tms, intsts = data[0], data[1]

        # add to plot
        fig.add_trace(Scatter(x=tms,
                              y=intsts,
                              name=ff_name,
                              line=dict(
                                  width=4,
                                  color=sim_colors[ff_name]
                                        )
                              )
                      )
        if averaging == 'slow_fast':
            try:
                data = sql.fetch_table_where2(db, f'{ff_name}__calcnoesy_fast', 'resid1', resid1, 'resid2',
                                              resid2).tms_intsts
            except:
                # for X-ray data
                data = sql.fetch_table_where2(db, f'{ff_name}__calcnoesy', 'resid1', resid1, 'resid2',
                                              resid2).tms_intsts

            # get data
            data = data.to_numpy()[0]
            tms, intsts = data[0], data[1]

            # add to plot
            fig.add_trace(Scatter(x=tms,
                                  y=intsts,
                                  name=f'{ff_name} (all fast)',
                                  line=dict(
                                      width=4,
                                      color=f'rgba{mcolors.to_rgba(sim_colors[ff_name], alpha=0.5)}'
                                            )
                                  )
                          )

    # layout
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')

    fig.update_layout(
        autosize=False,
        yaxis=dict(
            title_text="Intensities",
            range=[0, 1]
        ),
        xaxis=dict(
            title_text="tm [s]"
        )

    )

    return fig

# Index Page callback
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-2':
        return page_2.page_2_layout
    elif pathname == '/page-1':
        return page_1.page_1_layout
    else:
        return home.home_layout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--db",
                        default='db/calcnoesy-supplement-data.db',
                        help="path to SQL db")

    args = parser.parse_args()

    db = Path(ROOTDIR, args.db)

    # get names of simulations in db
    sim_names = get_sim_names()
    save_file('sim_names', sim_names)

    # get resids and slow_resid pairs
    for i, sim_name in enumerate(sim_names):
        if sql.is_table(db, f"{sim_name}__pair__PC1"):
            resids = sorted(set(sql.fetch_column(db, f'{sim_name}__calcnoesy', 'resid1').resid1))
            save_file('resids', resids)
            slow_pairs = sql.fetch_column_names(db, f"{sim_name}__pair__PC1")
            save_file('slow_pairs', slow_pairs)
            break

    ms_colors = {i: list(mcolors.TABLEAU_COLORS.values())[i] for i in range(10)}
    sim_colors = {key: list(mcolors.TABLEAU_COLORS.values())[i] for i, key in enumerate(sim_names)}

    # import pages
    from pages import page_1
    from pages import page_2
    from pages import home

    try:
        app.run_server(debug=True)
    except:
        import requests
        resp = requests.get('http://localhost:8050/shutdown')
        app.run_server(debug=True)