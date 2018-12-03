## How to run
## python hit_ratio.py predictions.txt /Users/arun/Downloads/datasets/yelp/PA/test/PA_test_yelp_academic_dataset_review.csv

##  Compute the mean average precision score for predictions

##  Prediction file format
##  user_id1,bus1,bus2,bus3,...
##  user_id2,bus2,bus4,bus5,...

## Validation file format
## review_test.csv file with headers -> user_id, business_id


import pandas as pd
import sys


def hit_ratio(prediction_map, evaluation_set):
    hit_ratio = 0
    count = 0
    for p in prediction_map:
        count += 1
        preds = prediction_map[p]
        correct_predictions = 0
        
        for pred in preds:
            if p in evaluation_set and pred in evaluation_set[p]:
                correct_predictions += 1

        if correct_predictions > 0:
            hits = float(correct_predictions) / len(evaluation_set[p])
            hit_ratio += hits
    
    return float(hit_ratio)/count


def hit_ratio2(prediction_map, evaluation_set):
    total_test_size = 0
    correct_predictions = 0
    for p in prediction_map:
        preds = prediction_map[p]

        for pred in preds:
            if p in evaluation_set and pred in evaluation_set[p]:
                correct_predictions += 1

        if p in evaluation_set:
            total_test_size += len(evaluation_set[p])

    return float(correct_predictions) / total_test_size


def main():
    predictions_file = sys.argv[1]
    evaluation_file = sys.argv[2]

    evaluations = pd.read_csv(evaluation_file)
    prediction_map = dict()
    evaluation_set = dict()

    with open(predictions_file, "r") as csv_file:
        for line in csv_file:
            preds = line.strip().split(",")
            user_id = preds[0]
            busses = preds[1:]

            prediction_map[user_id] = busses

    for row in evaluations[['user_id', 'business_id']].values:
        if not row[0] in evaluation_set:
            evaluation_set[row[0]] = set()
        
        evaluation_set[row[0]].add(row[1])

    hr = hit_ratio2(prediction_map, evaluation_set)

    print "HIT RATIO Score: ", hr


if __name__ == "__main__":
    main()
