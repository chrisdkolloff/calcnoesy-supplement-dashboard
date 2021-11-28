import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

home_layout = html.Div([
    dcc.Markdown('''
    # Supplementary Material: Effect of supra-\u03c4c conformational exchange on the NOESY build-up of large proteins
    ## Christopher Kolloff, Adam Mazur, Jan K. Marzinek, Peter J. Bond, Simon Olsson, and Sebastian Hiller
    ## DOI: [XXX](https://todo)
    '''),
    html.Br(),
    dcc.Markdown('''
    ### This dashboard contains all supra-\u03c4c tICs as well as the implied timescale of the slow processes on [page 1](/page-1). On [page 2](/page-2), you will find the NOESY build-ups between methyl groups calculated for the respective force fields.
    
    ### For questions or remarks, please reach out to [Christopher Kolloff](mailto:chrisdkolloff@gmail.com) or [Sebastian Hiller](mailto:sebastian.hiller@unibas.ch).
    '''),
    html.Br(),
    html.Div(id='home-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
])