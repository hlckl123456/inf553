Contains Jupyter notebook used to experiment with the Model Based approach approach.

Needs the data from Dataset/PA/Restaurants to run.

Will write the top 50 predictions for each user to als_predictions.txt

You can use hit_ratio.py to evaluate the results against a known test set.
Eg. python hit_ratio.py als_predictions.txt Dataset/PA/Restaurants/test/PA_test_yelp_academic_dataset_review.csv
