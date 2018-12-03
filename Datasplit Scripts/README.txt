File: jsontocsv.py
The jsontocsv.py script converts yelp dataset from the default JSON format into csv format

File: datasplit.py
The datasplit script splits the csv yelp dataset into csv containing only Pennsylvennia State PA
The line can be uncommented
to get other states as well by providing a state.txt file containing each state per line
#states = load_states()

File: projvalidate.py
This file validates the review count per business and creates a column called review_count2 in the business.csv
We were worried that the review count reported by each business were invalid as some states contained very
few data