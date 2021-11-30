# Supplementary Material: Effect of supra-τc conformational exchange on the NOESY build-up of large proteins

## Setup
After cloning the repository, create a new environment that will install the necessary dependencies using the .yml file
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

``` https://data.mendeley.com/XXX ```

After downloading the data, move the database into `/db/`

## Running the dashboard
The dashboard will run on a local server using:

```
python app.py ./db/tica.db
```

