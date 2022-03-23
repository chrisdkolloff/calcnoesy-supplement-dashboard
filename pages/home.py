from dash import dcc
from dash import html

home_layout = html.Div([
    dcc.Markdown('''
    # Supplementary Material: Motional clustering in supra-\u03c4c conformational exchange influences NOE cross-relaxation rate
    ## Christopher Kolloff, Adam Mazur, Jan K. Marzinek, Peter J. Bond, Simon Olsson, and Sebastian Hiller
    ## DOI: [10.1016/j.jmr.2022.107196](https://doi.org/10.1016/j.jmr.2022.107196)
    '''),
    html.Br(),
    dcc.Markdown('''
    ### This dashboard contains correlation functions, ITS and tICA plots as well as NOESY build-up curves for all local systems in all force fields that are in supra-\u03c4c exchange (up to 10 A distance). In addition, all other methyl--methyl contacts of up to 5 A are also provided. 
    ### [Page 1](/page-1) contains the ITS and tICA plots of the systems, on [page 2](/page-2), you will find all NOESY build-up curves between methyl groups calculated for the respective force fields, and [page 3](/page-3) contains the correlation functions of the local systems. 
    
    ### For questions or remarks, please reach out to [Christopher Kolloff](mailto:kolloff@chalmers.se) or [Sebastian Hiller](mailto:sebastian.hiller@unibas.ch).
    '''),
    html.Br(),
    html.Div(id='home-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go to Page 3', href='/page-3'),
])