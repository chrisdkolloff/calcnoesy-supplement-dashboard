# Supplementary Material: Effect of supra-τc conformational exchange on the NOESY build-up of large proteins

## Setup
After cloning the repository with

```
$ git clone git@github.com:chrisdkolloff/calcnoesy-supplement-dashboard.git
```

create a new environment that will install the necessary dependencies using the .yml file
provided:

```
$ conda env create -f environment.yml
```

### Dependencies
Alternatively, install the following dependencies:
```
python=3.8
pyemma
dash
dash-core-components
dash-html-components
pandas
```

## Data
Download the SQL database containing the data used in this study using the following link:

``` 
https://data.mendeley.com/XXX
DOI: 10.17632/d9r96x4c9x.1
```

After downloading the data, move the database into `/db/`

## Running the dashboard
The dashboard will run on a local server using:

```
python app.py
```

After successful execution, you will see the following:

```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
```
You can then open the app by copy-pasting the address ```http://127.0.0.1:8050/``` into your browser. 