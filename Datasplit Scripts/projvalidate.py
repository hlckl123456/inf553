import pyspark
from pyspark.sql import SQLContext
import pandas as pd
import csv
import os


def load_states():
    # read US states
    f = open('states.txt', 'r')

    states = set()
    for line in f.readlines():
        l = line.strip('\n')
        if l != '':
            states.add(l)

    return states


def validate2(states, bt):
    #sqlContext = SQLContext(sc)

    for state in states:
        if not os.path.exists("US/" + state):
            continue
        """
        Train
        """

        train_prefix = "US/" + state + '/' + bt + "/train/" + state + "_train_"

        business_train_fname = train_prefix + 'yelp_academic_dataset_business.csv'
        business_train_fname2 = train_prefix + 'yelp_academic_dataset_business2.csv'
        review_train_fname = train_prefix + 'yelp_academic_dataset_review.csv'
        checkins_train_fname = train_prefix + 'yelp_academic_dataset_checkin.csv'
        tip_train_fname = train_prefix + 'yelp_academic_dataset_tip.csv'
        user_train_fname = train_prefix + 'yelp_academic_dataset_user.csv'

        df_business_train = pd.read_csv(business_train_fname)
        df_review_train = pd.read_csv(review_train_fname)
        df_checkins_train = pd.read_csv(checkins_train_fname)
        df_tip_train = pd.read_csv(tip_train_fname)
        df_user_train = pd.read_csv(user_train_fname)

        count_business_train = df_business_train.shape[0]
        count_review_train = df_review_train.shape[0]
        count_checkins_train = df_checkins_train.shape[0]
        count_tip_train = df_tip_train.shape[0]
        count_user_train = df_user_train.shape[0]

        df_train_busi_review_count = df_review_train.groupby(['business_id']).agg(['count'])
        dict_train_busi_review_count = df_train_busi_review_count['review_id'].apply(list).to_dict()['count']
        new_pdf_train_busi_review_count = pd.DataFrame.from_dict(dict_train_busi_review_count, orient='index').reset_index()
        new_pdf_train_busi_review_count.columns = ['business_id', 'review_count2']

        df_business_train = df_business_train.join(new_pdf_train_busi_review_count.set_index('business_id'), on='business_id')

        df_business_train.to_csv(business_train_fname2, index=False)

        """
        Test
        """
        valid_prefix = "US/" + state + '/' + bt + "/valid/" + state + "_valid_"

        business_valid_fname = valid_prefix + 'yelp_academic_dataset_business.csv'
        business_valid_fname2 = valid_prefix + 'yelp_academic_dataset_business2.csv'
        review_valid_fname = valid_prefix + 'yelp_academic_dataset_review.csv'
        checkins_valid_fname = valid_prefix + 'yelp_academic_dataset_checkin.csv'
        tip_valid_fname = valid_prefix + 'yelp_academic_dataset_tip.csv'
        user_valid_fname = valid_prefix + 'yelp_academic_dataset_user.csv'

        df_business_valid = pd.read_csv(business_valid_fname)
        df_review_valid = pd.read_csv(review_valid_fname)
        df_checkins_valid = pd.read_csv(checkins_valid_fname)
        df_tip_valid = pd.read_csv(tip_valid_fname)
        df_user_valid = pd.read_csv(user_valid_fname)

        count_business_valid = df_business_valid.shape[0]
        count_review_valid = df_review_valid.shape[0]
        count_checkins_valid = df_checkins_valid.shape[0]
        count_tip_valid = df_tip_valid.shape[0]
        count_user_valid = df_user_valid.shape[0]

        df_valid_busi_review_count = df_review_valid.groupby(['business_id']).agg(['count'])
        dict_valid_busi_review_count = df_valid_busi_review_count['review_id'].apply(list).to_dict()['count']
        new_pdf_valid_busi_review_count = pd.DataFrame.from_dict(dict_valid_busi_review_count, orient='index').reset_index()
        new_pdf_valid_busi_review_count.columns = ['business_id', 'review_count2']

        df_business_valid = df_business_valid.join(new_pdf_valid_busi_review_count.set_index('business_id'), on='business_id')

        df_business_valid.to_csv(business_valid_fname2, index=False)

        """
        Test
        """
        test_prefix = "US/" + state + '/' + bt + "/test/" + state + "_test_"

        business_test_fname = test_prefix + 'yelp_academic_dataset_business.csv'
        business_test_fname2 = test_prefix + 'yelp_academic_dataset_business2.csv'
        review_test_fname = test_prefix + 'yelp_academic_dataset_review.csv'
        checkins_test_fname = test_prefix + 'yelp_academic_dataset_checkin.csv'
        tip_test_fname = test_prefix + 'yelp_academic_dataset_tip.csv'
        user_test_fname = test_prefix + 'yelp_academic_dataset_user.csv'

        df_business_test = pd.read_csv(business_test_fname)
        df_review_test = pd.read_csv(review_test_fname)
        df_checkins_test = pd.read_csv(checkins_test_fname)
        df_tip_test = pd.read_csv(tip_test_fname)
        df_user_test = pd.read_csv(user_test_fname)

        count_business_test = df_business_test.shape[0]
        count_review_test = df_review_test.shape[0]
        count_checkins_test = df_checkins_test.shape[0]
        count_tip_test = df_tip_test.shape[0]
        count_user_test = df_user_test.shape[0]

        df_test_busi_review_count = df_review_test.groupby(['business_id']).agg(['count'])
        dict_test_busi_review_count = df_test_busi_review_count['review_id'].apply(list).to_dict()['count']
        new_pdf_test_busi_review_count = pd.DataFrame.from_dict(dict_test_busi_review_count, orient='index').reset_index()
        new_pdf_test_busi_review_count.columns = ['business_id', 'review_count2']

        df_business_test = df_business_test.join(new_pdf_test_busi_review_count.set_index('business_id'), on='business_id')

        df_business_test.to_csv(business_test_fname2, index=False)

        # write other info to csv
        with open("US/" + state + '/' + bt + '/' + state + '_stats.csv', mode='wb') as f:
            writer = csv.writer(f)

            writer.writerow(["Business Train Count", count_business_train])
            writer.writerow(["Review Train Count", count_review_train])
            writer.writerow(["Check-in Train Count", count_checkins_train])
            writer.writerow(["Tip Train Count", count_tip_train])
            writer.writerow(["User Train Count", count_user_train])
            
            writer.writerow(["Business valid Count", count_business_valid])
            writer.writerow(["Review valid Count", count_review_valid])
            writer.writerow(["Check-in valid Count", count_checkins_valid])
            writer.writerow(["Tip valid Count", count_tip_valid])
            writer.writerow(["User valid Count", count_user_valid])
            
            writer.writerow(["Business Test Count", count_business_test])
            writer.writerow(["Review Test Count", count_review_test])
            writer.writerow(["Check-in Test Count", count_checkins_test])
            writer.writerow(["Tip Test Count", count_tip_test])
            writer.writerow(["User Test Count", count_user_test])

    return


def validate(states):
    # sqlContext = SQLContext(sc)

    for state in states:
        if not os.path.exists("US/" + state):
            continue
        """
        Train
        """

        train_prefix = "US/" + state + "/train/" + state + "_train_"

        business_train_fname = train_prefix + 'yelp_academic_dataset_business.csv'
        business_train_fname2 = train_prefix + 'yelp_academic_dataset_business2.csv'
        review_train_fname = train_prefix + 'yelp_academic_dataset_review.csv'
        checkins_train_fname = train_prefix + 'yelp_academic_dataset_checkin.csv'
        tip_train_fname = train_prefix + 'yelp_academic_dataset_tip.csv'
        user_train_fname = train_prefix + 'yelp_academic_dataset_user.csv'

        df_business_train = pd.read_csv(business_train_fname)
        df_review_train = pd.read_csv(review_train_fname)
        df_checkins_train = pd.read_csv(checkins_train_fname)
        df_tip_train = pd.read_csv(tip_train_fname)
        df_user_train = pd.read_csv(user_train_fname)

        count_business_train = df_business_train.shape[0]
        count_review_train = df_review_train.shape[0]
        count_checkins_train = df_checkins_train.shape[0]
        count_tip_train = df_tip_train.shape[0]
        count_user_train = df_user_train.shape[0]

        df_train_busi_review_count = df_review_train.groupby(['business_id']).agg(['count'])
        dict_train_busi_review_count = df_train_busi_review_count['review_id'].apply(list).to_dict()['count']
        new_pdf_train_busi_review_count = pd.DataFrame.from_dict(dict_train_busi_review_count,
                                                                 orient='index').reset_index()
        new_pdf_train_busi_review_count.columns = ['business_id', 'review_count2']

        df_business_train = df_business_train.join(new_pdf_train_busi_review_count.set_index('business_id'),
                                                   on='business_id')

        df_business_train.to_csv(business_train_fname2, index=False)

        """
        Test
        """
        test_prefix = "US/" + state + "/test/" + state + "_test_"

        business_test_fname = test_prefix + 'yelp_academic_dataset_business.csv'
        business_test_fname2 = test_prefix + 'yelp_academic_dataset_business2.csv'
        review_test_fname = test_prefix + 'yelp_academic_dataset_review.csv'
        checkins_test_fname = test_prefix + 'yelp_academic_dataset_checkin.csv'
        tip_test_fname = test_prefix + 'yelp_academic_dataset_tip.csv'
        user_test_fname = test_prefix + 'yelp_academic_dataset_user.csv'

        df_business_test = pd.read_csv(business_test_fname)
        df_review_test = pd.read_csv(review_test_fname)
        df_checkins_test = pd.read_csv(checkins_test_fname)
        df_tip_test = pd.read_csv(tip_test_fname)
        df_user_test = pd.read_csv(user_test_fname)

        count_business_test = df_business_test.shape[0]
        count_review_test = df_review_test.shape[0]
        count_checkins_test = df_checkins_test.shape[0]
        count_tip_test = df_tip_test.shape[0]
        count_user_test = df_user_test.shape[0]

        df_test_busi_review_count = df_review_test.groupby(['business_id']).agg(['count'])
        dict_test_busi_review_count = df_test_busi_review_count['review_id'].apply(list).to_dict()['count']
        new_pdf_test_busi_review_count = pd.DataFrame.from_dict(dict_test_busi_review_count,
                                                                orient='index').reset_index()
        new_pdf_test_busi_review_count.columns = ['business_id', 'review_count2']

        df_business_test = df_business_test.join(new_pdf_test_busi_review_count.set_index('business_id'),
                                                 on='business_id')

        df_business_test.to_csv(business_test_fname2, index=False)

        # write other info to csv
        with open("US/" + state + '/' + state + '_stats.csv', mode='wb') as f:
            writer = csv.writer(f)

            writer.writerow(["Business Train Count", count_business_train])
            writer.writerow(["Review Train Count", count_review_train])
            writer.writerow(["Check-in Train Count", count_checkins_train])
            writer.writerow(["Tip Train Count", count_tip_train])
            writer.writerow(["User Train Count", count_user_train])

            writer.writerow(["Business Test Count", count_business_test])
            writer.writerow(["Review Test Count", count_review_test])
            writer.writerow(["Check-in Test Count", count_checkins_test])
            writer.writerow(["Tip Test Count", count_tip_test])
            writer.writerow(["User Test Count", count_user_test])

    return

def main():
    #sc = pyspark.SparkContext('local[2]')

    #states = load_states()

    states =set()
    states.add('PA')
    validate2(states, 'Restaurants')

    return


if __name__ == '__main__':
    main()
