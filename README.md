# CMPT-353 Project

Project Code for CMPT 353 Summer 2019. 
Project description can be found: https://coursys.sfu.ca/2019su-cmpt-353-d1/pages/Project

## WikiData Movie
Description: Data analysis and machine learning on wikidata movie data.

Objective: Predict audience average ratings
## Required Libraries
*  sys
*  pandas
*  numpy
*  sklearn
*  matplolib
*  scipy
*  statsmodels

## Usage

The first thing you should run:
```bash
$ python3 preproc.py
```
This will import the data and hopefully create a CSV file `ml.csv`.

The remainder of the files can be run in any order

**Machine Learning:** Extracts `ml.csv` into a dataframe and does ML analysis to try
predict audience averages. resutls are printed to the screen.
```bash
$ python3 ml.py
```
**Graphs:** Extracts `ml.csv` into a dataframe and saves charts and graphs to the
directory. Images are displayed in the report. You can optionally comment out
the `plt.savefig()` lie and use `plt.show()`.
```bash
$ python3 graph.py
```
**Stats:** Extracts `omdb-data.json.gz`, `rotten-tomatoes.json.gz`, `wikidata-movies.json.gz`, 
`genres.json.gz` into a dataframe and saves results from statistical analysis into 
`genre_pvalue.csv`, `actor_pvalue.csv`, and `director_pvalue.csv`
```bash
$ python3 stats2.py
```