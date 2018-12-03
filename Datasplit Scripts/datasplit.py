import pandas as pd
import numpy as np
import os


def load_states():
    # read US states
    f = open('US Provinces.txt', 'r')

    states = set()
    for line in f.readlines():
        l = line.strip('\n')
        if l != '':
            states.add(l)

    return states


def splitandwrite(df, path, st):
    business = 'yelp_academic_dataset_business.csv'
    review = 'yelp_academic_dataset_review.csv'
    checkins = 'yelp_academic_dataset_checkin.csv'
    tip = 'yelp_academic_dataset_tip.csv'
    user = 'yelp_academic_dataset_user.csv'

    # make train/test path if they don't exist
    path_train = path + 'train'
    if not os.path.exists(path_train):
        os.mkdir(path_train)

    path_test = path + 'test'
    if not os.path.exists(path_test):
        os.mkdir(path_test)

    # cross lookup on review dataset and split by 80/20
    review_df = pd.read_csv(review)

    # filter review by states
    review_df = review_df[review_df['business_id'].isin(df['business_id'])]

    # find 80 percentile
    date_column = list(review_df.sort_values('date')['date'])
    index = range(0, len(date_column) + 1)
    cut = date_column[np.int((np.percentile(index, 80)))]

    """
    Section Review
    """
    # cut by date
    review_train = review_df[review_df['date'] < cut]
    review_test = review_df[review_df['date'] >= cut]

    # write to train
    review_train.to_csv(path_train + '/' + st + '_train_' + review, index=False)
    # write to test
    review_test.to_csv(path_test + '/' + st + '_test_' + review, index=False)

    """
    Section Business
    """
    busi_train = df[df['business_id'].isin(review_train['business_id'])]
    busi_test = df[df['business_id'].isin(review_test['business_id'])]

    # write to train
    busi_train.to_csv(path_train + '/' + st + '_train_' + business, index=False)
    # write to test
    busi_test.to_csv(path_test + '/' + st + '_test_' + business, index=False)

    """
    Section Checkin
    """
    checkindf = pd.read_csv(checkins)
    checkin_train = checkindf[checkindf['business_id'].isin(busi_train['business_id'])]
    checkin_test = checkindf[checkindf['business_id'].isin(busi_test['business_id'])]

    # write to train
    checkin_train.to_csv(path_train + '/' + st + '_train_' + checkins, index=False)
    # write to test
    checkin_test.to_csv(path_test + '/' + st + '_test_' + checkins, index=False)

    """
    Section User
    """
    userdf = pd.read_csv(user)
    user_train = userdf[userdf['user_id'].isin(review_train['user_id'])]
    user_test = userdf[userdf['user_id'].isin(review_test['user_id'])]

    # write to train
    user_train.to_csv(path_train + '/' + st + '_train_' + user, index=False)
    # write to test
    user_test.to_csv(path_test + '/' + st + '_test_' + user, index=False)

    """
    Section Tip
    """
    tipdf = pd.read_csv(tip)
    tip_train = tipdf[tipdf['user_id'].isin(user_train['user_id'])]
    tip_train = tip_train[tip_train['business_id'].isin(busi_train['business_id'])]
    tip_test = tipdf[tipdf['user_id'].isin(user_test['user_id'])]
    tip_test = tip_test[tip_test['business_id'].isin(busi_test['business_id'])]

    # write to train
    tip_train.to_csv(path_train + '/' + st + '_train_' + tip, index=False)
    # write to test
    tip_test.to_csv(path_test + '/' + st + '_test_' + tip, index=False)

def splitandwrite2(df, path, st, bt):
    business = 'yelp_academic_dataset_business.csv'
    review = 'yelp_academic_dataset_review.csv'
    checkins = 'yelp_academic_dataset_checkin.csv'
    tip = 'yelp_academic_dataset_tip.csv'
    user = 'yelp_academic_dataset_user.csv'

    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(path + bt):
        os.mkdir(path + bt)

    # make train/test path if they don't exist
    path_train = path + bt + '/train'
    if not os.path.exists(path_train):
        os.mkdir(path_train)

    path_test = path + bt + '/test'
    if not os.path.exists(path_test):
        os.mkdir(path_test)

    path_valid = path + bt + '/valid'
    if not os.path.exists(path_valid):
        os.mkdir(path_valid)

    # cross lookup on review dataset and split by 80/10/10
    review_df = pd.read_csv(review)

    # filter review by states
    review_df = review_df[review_df['business_id'].isin(df['business_id'])]

    # find 80 percentile
    date_column = list(review_df.sort_values('date')['date'])
    index = range(0, len(date_column) + 1)
    cut1 = 80
    cut2 = 90

    # change date to int
    review_df["date"] = pd.to_datetime(review_df["date"]).dt.strftime("%Y%m%d").astype(int)
    review_df["1overN"] = review_df.groupby(["user_id"])['date'].transform('count')
    review_df["1overN"] = review_df['1overN'].apply(lambda x: 1.0 / x * 100)
    review_df["2overN"] = review_df['1overN'].apply(lambda x: 100 - 2 * x)
    review_df["1overN"] = review_df['1overN'].apply(lambda x: 100 - x)

    """
    Section Review
    """
    # cut by date
    df_perct = review_df.assign(percentile=review_df.groupby("user_id")['date'].rank(pct=True).mul(100))

    # test
    review_test = df_perct[df_perct.apply(lambda x: x["percentile"] > x["1overN"] or x["percentile"] > cut2, axis=1)]
    review_valid = df_perct[df_perct.apply(lambda x: x["percentile"] <= min(x["1overN"], cut2) and (x["percentile"] > x["2overN"] or x["percentile"] > cut1), axis=1)]
    review_train = df_perct[df_perct.apply(lambda x: x["percentile"] <= min(x["2overN"], cut1), axis=1)]

    # write to train
    review_train.to_csv(path_train + '/' + st + '_train_' + review, index=False)
    # write to valid
    review_valid.to_csv(path_valid + '/' + st + '_valid_' + review, index=False)
    # write to test
    review_test.to_csv(path_test + '/' + st + '_test_' + review, index=False)

    """
    Section Business
    """
    busi_train = df[df['business_id'].isin(review_train['business_id'])]
    busi_valid = df[df['business_id'].isin(review_valid['business_id'])]
    busi_test = df[df['business_id'].isin(review_test['business_id'])]

    # write to train
    busi_train.to_csv(path_train + '/' + st + '_train_' + business, index=False)
    # write to valid
    busi_valid.to_csv(path_valid + '/' + st + '_valid_' + business, index=False)
    # write to test
    busi_test.to_csv(path_test + '/' + st + '_test_' + business, index=False)

    """
    Section Checkin
    """
    checkindf = pd.read_csv(checkins)
    checkin_train = checkindf[checkindf['business_id'].isin(busi_train['business_id'])]
    checkin_valid = checkindf[checkindf['business_id'].isin(busi_valid['business_id'])]
    checkin_test = checkindf[checkindf['business_id'].isin(busi_test['business_id'])]

    # write to train
    checkin_train.to_csv(path_train + '/' + st + '_train_' + checkins, index=False)
    # write to valid
    checkin_valid.to_csv(path_valid + '/' + st + '_valid_' + checkins, index=False)
    # write to test
    checkin_test.to_csv(path_test + '/' + st + '_test_' + checkins, index=False)

    """
    Section User
    """
    userdf = pd.read_csv(user)
    user_train = userdf[userdf['user_id'].isin(review_train['user_id'])]
    user_valid = userdf[userdf['user_id'].isin(review_valid['user_id'])]
    user_test = userdf[userdf['user_id'].isin(review_test['user_id'])]

    # write to train
    user_train.to_csv(path_train + '/' + st + '_train_' + user, index=False)
    # write to valid
    user_valid.to_csv(path_valid + '/' + st + '_valid_' + user, index=False)
    # write to test
    user_test.to_csv(path_test + '/' + st + '_test_' + user, index=False)

    """
    Section Tip
    """
    tipdf = pd.read_csv(tip)
    tip_train = tipdf[tipdf['user_id'].isin(user_train['user_id'])]
    tip_train = tip_train[tip_train['business_id'].isin(busi_train['business_id'])]
    tip_valid = tipdf[tipdf['user_id'].isin(user_valid['user_id'])]
    tip_valid = tip_valid[tip_valid['business_id'].isin(busi_valid['business_id'])]
    tip_test = tipdf[tipdf['user_id'].isin(user_test['user_id'])]
    tip_test = tip_test[tip_test['business_id'].isin(busi_test['business_id'])]

    # write to train
    tip_train.to_csv(path_train + '/' + st + '_train_' + tip, index=False)
    # write to valid
    tip_valid.to_csv(path_valid + '/' + st + '_valid_' + tip, index=False)
    # write to test
    tip_test.to_csv(path_test + '/' + st + '_test_' + tip, index=False)


def country_split(states):
    fname = 'yelp_academic_dataset_business.csv'

    df = pd.read_csv(fname)

    # US
    df_us = df[df['state'].isin(states)]
    st = df_us['state'].unique()
    if not os.path.exists('US'):
        os.mkdir('US')

    for s in st:
        df_us_s = df_us[df_us['state'] == s]

        p1 = 'US/' + s
        if not os.path.exists(p1):
            os.mkdir(p1)

        splitandwrite(df_us_s, p1 + '/', s)

    # CA
    df_ca = df[~df['state'].isin(states)]
    pro = df_ca['state'].unique()
    if not os.path.exists('CA'):
        os.mkdir('CA')

    for s in pro:
        df_ca_s = df_ca[df_ca['state'] == s]

        p2 = 'CA/' + s
        if not os.path.exists(p2):
            os.mkdir(p2)

        splitandwrite(df_ca_s, p2 + '/', s)


def country_split2(states):
    fname = 'yelp_academic_dataset_business.csv'

    df = pd.read_csv(fname)

    # US
    df_us = df[df['state'].isin(states)]
    st = df_us['state'].unique()
    if not os.path.exists('US'):
        os.mkdir('US')


    for s in st:
        df_us_s = df_us[df_us['state'] == s]

        p1 = 'US/' + s
        if not os.path.exists(p1):
            os.mkdir(p1)

        # find all business type

        bt = 'Restaurants'
        df_us_s_bt = df_us_s[df_us_s['categories'].str.contains(bt, na=False)]

        splitandwrite2(df_us_s_bt, p1 + '/', s, bt)

def main():

    #states = load_states()

    states = set()
    states.add('PA')

    country_split2(states)

    return


if __name__ == '__main__':
    main()
