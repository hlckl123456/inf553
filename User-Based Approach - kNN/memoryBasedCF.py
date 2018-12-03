import re
import os
import copy
import nltk
import shutil
import numpy as np
import random
from nltk.corpus import stopwords
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType, DoubleType
from pyspark.sql import SQLContext
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


WORD = re.compile(r'\w+')
user_user_map = {}

# https://stackoverflow.com/questions/15173225/calculate-cosine-similarity-given-2-sentence-strings
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def tfidf(text, transformer, lshf, k, mapping):
    ts = transformer.transform([text])
    scores, idx = lshf.kneighbors(ts, n_neighbors=k)

    # return mappings with user id
    return scores[0], [mapping[x] for x in idx[0]]


def strjoin(strx, strjoined=""):
    return strjoined + " " + strx


def filterStopwords(s, stop_words):
    sb = ""
    for word in s.split(" "):
        if word.lower() not in stop_words:
            sb += word + " "
    return sb


def weightedVote(nn, userBusiMap, busiAttrMap):
    nn_users = nn[1]
    nn_similarity = nn[0]

    s_random = None
    s_goodfor = None
    s_star = -1
    s_alc = -1
    s_noise = -1
    s_prange = -1
    weight_sum = 0.0
    # find weighted average from all users
    for i in range(len(nn_users)):
        # find the weighted similarity score
        similarity = 1 - nn_similarity[i]
        weight = (len(nn_users) - 1 - i) / (len(nn_users) - 1)
        weighted_similarity = similarity * weight
        weight_sum += weighted_similarity

        B = float(len(userBusiMap[str(nn_users[i])]))
        u_random = None
        u_goodfor = None
        u_star = -1
        u_alc = -1
        u_noise = -1
        u_prange = -1

        # find business average from that user
        for busi in userBusiMap[str(nn_users[i])]:

            busi = str(busi)

            attr = busiAttrMap[busi]
            # random attr
            if u_random is None:
                u_random = copy.deepcopy(attr[0])
            else:
                u_random += attr[0]
            # good for attr
            if u_goodfor is None:
                u_goodfor = copy.deepcopy(attr[1])
            else:
                u_goodfor += attr[1]
            # star attr
            if u_star == -1:
                u_star = attr[2]
            else:
                u_star += attr[2]
            # alc attr
            if u_alc == -1:
                u_alc = attr[3]
            else:
                u_alc += attr[3]
            # noice attr
            if u_noise == -1:
                u_noise = attr[4]
            else:
                u_noise += attr[4]
            # prange attr
            if u_prange == -1:
                u_prange = attr[5]
            else:
                u_prange += attr[5]

        u_random = (u_random / B) * weighted_similarity
        u_goodfor = (u_goodfor / B) * weighted_similarity
        u_star = (u_star / B) * weighted_similarity
        u_alc = (u_alc / B) * weighted_similarity
        u_noise = (u_noise / B) * weighted_similarity
        u_prange = (u_prange / B) * weighted_similarity

        # random attr
        if s_random is None:
            s_random = copy.deepcopy(u_random)
        else:
            s_random += u_random
        # good for attr
        if s_goodfor is None:
            s_goodfor = copy.deepcopy(u_goodfor)
        else:
            s_goodfor += u_goodfor
        # star
        if s_star == -1:
            s_star = u_star
        else:
            s_star += u_star
        # alc attr
        if s_alc == -1:
            s_alc = u_alc
        else:
            s_alc += u_alc
        # noice attr
        if s_noise == -1:
            s_noise = u_noise
        else:
            s_noise += u_noise
        # prange attr
        if s_prange == -1:
            s_prange = u_prange
        else:
            s_prange += u_prange

    s_random /= weight_sum
    s_goodfor /= weight_sum
    s_star /= weight_sum
    s_alc /= weight_sum
    s_noise /= weight_sum
    s_prange /= weight_sum

    return s_random, s_goodfor, s_star, s_alc, s_noise, s_prange


def getSimilarity(uid1, uid2, vec1, vec2):
    A = uid1
    B = uid2

    if A > B:
        A = uid2
        B = uid1

    if (A, B) in user_user_map:
        return user_user_map[(A, B)]
    else:
        cs = cosine_similarity(vec1, vec2)[0][0]
        user_user_map[(A, B)] = cs

    return cs


def findTopN(uid, text, rdd):
    l = rdd.map(lambda x: (x[0], cosine_similarity(uid, x[0], text, x[1]))).takeOrdered(10, key=lambda x: -x[1])
    return l


def toVec(arr):
    arr = [random.choice([True, False]) if v is None else v for v in arr]
    return np.array(arr, dtype=float)


def alcholToScaler(a):
    if str(a) == 'full_bar':
        return 3.0
    elif str(a) == 'beer_and_wine':
        return 2.0
    elif str(a) == 'none':
        return 1.0
    else:
        return 0.0


def noiseLevelToScaler(a):
    if str(a) == 'very_loud':
        return 4.0
    elif str(a) == 'loud':
        return 3.0
    elif str(a) == 'average':
        return 2.0
    elif str(a) == 'quite':
        return 1.0
    else:
        return 0.0


def fillPrange(prange):
    if prange == None:
        return 1.0
    else:
        return prange


def findNN(rawData_review):
    # fetch user_id and review text + xstar
    data_review = rawData_review.map(lambda x: (x[1], x[3] + " " + str(x[5]) + "star"))

    # knn parameter
    k = 10

    # nltk
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    # find top N users similar to user
    # go through train_review and generate word cloud for each user
    userReviewText = data_review.reduceByKey(strjoin)
    userReviewText = userReviewText.map(lambda x: (x[0], filterStopwords(x[1], stop_words)))

    # find corpus
    # if os.path.exists("feature.pkl"):
    #    # Load it
    #    transformer = pickle.load(open("transformer.pkl", "rb"))
    #    features = pickle.load(open("feature.pkl", "rb"))
    # else:
    # (id, text)
    corpusFullMap = userReviewText.map(lambda x: (x[0], x[1])).collect()
    corpusFullMap = zip(*corpusFullMap)

    # get the partial map of (idx, id) in the form of dict and (text)
    idMap = dict(enumerate(corpusFullMap[0]))
    textMap = corpusFullMap[1]

    transformer = TfidfVectorizer(min_df=20, ngram_range=(1, 3), analyzer='word', max_features=1000)
    features = transformer.fit_transform(textMap)
    # Save transformer
    # with open('transformer.pkl', 'wb') as f:
    #    pickle.dump(transformer, f, pickle.HIGHEST_PROTOCOL)
    # Save features
    # with open('feature.pkl', 'wb') as f:
    #    pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

    # initialize lsh
    lshf = LSHForest(random_state=42)
    lshf.fit(features)

    # find tf-idf of word vector
    userReviewVec = userReviewText.map(lambda x: (x[0], tfidf(x[1], transformer, lshf, k, idMap)))

    # test load and save of rdd

    # if os.path.exists("result"):
    #    shutil.rmtree("result")

    # userReviewVec.map(lambda x: (str(x[0]), " ".join(map(str, x[1][0])), " ".join(map(str, x[1][1]))))\
    #    .saveAsTextFile('result')
    return userReviewVec, lshf, transformer, idMap, stop_words


def findBusiAttrMapping(rawData_business):
    data_business = rawData_business.map(lambda x: (x[82], ([x[4], x[20], x[22], x[25], x[60], x[73]]),
                                                    ([x[45], x[46], x[47], x[48], x[49], x[50]]), x[99], x[3], x[68], x[75]))

    data_business = data_business.map(lambda x: (str(x[0]), (toVec(x[1]), toVec(x[2]), x[3], alcholToScaler(x[4]), noiseLevelToScaler(x[5]), fillPrange(x[6]))))

    return data_business.collectAsMap()


def computeMSE(vec1, vec2):
    # 15 attributes in total, 6 random, 6 good for, 3 non binary
    rand = list(np.abs(vec1[0] - vec2[0]) ** 2)
    gf = list(np.abs(vec1[1] - vec2[1]) ** 2)
    star = list([np.abs(vec1[2] - vec2[2]) ** 2])
    alc = list([np.abs(vec1[3] - vec2[3]) ** 2])
    noise = list([np.abs(vec1[4] - vec2[4]) ** 2])
    prange = list([np.abs(vec1[5] - vec2[5]) ** 2])

    error = []
    error.extend(rand)
    error.extend(gf)
    error.extend(star)
    error.extend(alc)
    error.extend(noise)
    error.extend(prange)

    return np.array(error)


def writeToJson(a):
    fout = "file.txt"
    fo = open(fout, "wb")

    for k, v in a.items():
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')

    fo.close()


def collapse(x1, x2, x3, x4, x5, x6):
    res = []
    res.extend(x1)
    res.extend(x2)
    res.extend([x3, x4, x5, x6])
    return res


def businessRank(vec, neighbors, keys):
    distances, indices = neighbors.kneighbors([vec])
    return keys[indices][0]


def countAppearance(arr1, arr2):
    count = []
    for business in arr1:
        if business in arr2:
            count.append(1)
        else:
            count.append(0)
    return count


def runPrediction(sc, sqlContext, reviewPath, businessPath, schema_review, userBusiMap, busiAttrMap, lshf, transformer, idMap, stop_words, type):
    # read the business and review data

    rawData_review = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .schema(schema_review) \
        .load(reviewPath).rdd

    rawData_business = sqlContext.read.format("com.databricks.spark.csv")\
        .option("header", "true")\
        .option("inferschema", "true")\
        .option("mode", "DROPMALFORMED")\
        .load(businessPath).rdd

    # filter out user id who are not seen in the train user-review set
    rawData_review = rawData_review.filter(lambda x: str(x[1]) in userBusiMap)
    data_review = rawData_review.map(lambda x: (x[1], x[3] + " " + str(x[5]) + "star"))

    # knn parameter
    k = 10

    userReviewText = data_review.reduceByKey(strjoin)
    userReviewText = userReviewText.map(lambda x: (x[0], filterStopwords(x[1], stop_words)))

    # find tf-idf of word vector
    userReviewVec = userReviewText.map(lambda x: (x[0], tfidf(x[1], transformer, lshf, k, idMap)))

    # true business attribute mapping
    busiAttrMap2 = findBusiAttrMapping(rawData_business)

    userReviewVec = userReviewVec.map(lambda x: (x[0], weightedVote(x[1], userBusiMap, busiAttrMap)))

    true_review = rawData_review.map(lambda x: (x[1], busiAttrMap2[str(x[4])])).collectAsMap()

    result = userReviewVec.filter(lambda x: x[0] in true_review).map(lambda x: (x[0], (x[1], true_review[x[0]])))

    MSE = result.map(lambda x: computeMSE(x[1][0], x[1][1])).collect()
    MSE = np.mean(MSE, axis=0)
    RMSE = MSE ** 0.5

    print RMSE

    with open('result_' + type + '.txt', 'w') as f:
        f.writelines([str(RMSE)])

    # save prediction results in folder
    result = result.sortByKey()
    if os.path.exists('result/' + type + '_result'):
        shutil.rmtree('result/' + type + '_result')
    result.saveAsTextFile('result/' + type + '_result')

    # find top 50 similar business and compute hit ratio
    user_prediction = result.map(lambda x: (x[0], collapse(x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5])))
    business_flattened = {}
    for key, value in busiAttrMap.items():
        business_flattened[key] = collapse(value[0], value[1], value[2], value[3], value[4], value[5])
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='cosine').fit(np.array(business_flattened.values()))
    nbrs_keys = np.array(business_flattened.keys())

    # finds the top 50 matches
    top50 = user_prediction.map(lambda x: (x[0], businessRank(x[1], nbrs, nbrs_keys)))

    # save the result
    if os.path.exists('result/' + type + '_top50'):
        shutil.rmtree('result/' + type + '_top50')
    top50.saveAsTextFile('result/' + type + '_top50')

    # use the ranking algorithm find the hit
    hit = top50.flatMap(lambda x: countAppearance(userBusiMap[x[0]], x[1])).collect()

    # save the hit ratio to txt
    hit_ratios = np.mean(hit)
    with open('hit_ratio_' + type + '.txt', 'w') as f:
        f.writelines([str(hit_ratios)])


def main():
    random.seed(2018)

    # spark config
    conf = SparkConf()
    conf.setMaster("local").setAppName("MemoryBasedCF")
    conf.set("spark.network.timeout", "3600s")
    conf.set("spark.executor.heartbeatInterval", "3000s")

    sc = SparkContext(conf=conf)
    #sc.setLogLevel("ERROR")
    sc.setCheckpointDir("checkpoint")
    sqlContext = SQLContext(sc)

    '''
    load train data
    '''
    train_path = 'PA/Restaurants/train/'
    train_user = train_path + 'PA_train_yelp_academic_dataset_user.csv'
    train_review = train_path + 'PA_train_yelp_academic_dataset_review.csv'
    train_business = train_path + 'PA_train_yelp_academic_dataset_business.csv'
    train_tips = train_path + 'PA_train_yelp_academic_dataset_tip.csv'
    train_checkin = train_path + 'PA_train_yelp_academic_dataset_checkin.csv'

    schema_review = StructType([
        StructField("funny", IntegerType()),
        StructField("user_id", StringType()),
        StructField("review_id", StringType()),
        StructField("text", StringType()),
        StructField("business_id", StringType()),
        StructField("stars", IntegerType()),
        StructField("date", StringType()),
        StructField("useful", IntegerType()),
        StructField("cool", IntegerType()),
        StructField("1overN", DoubleType()),
        StructField("2overN", DoubleType()),
        StructField("percentile", DoubleType())
    ])

    rawData_review = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .schema(schema_review) \
        .load(train_review).rdd

    rawData_business = sqlContext.read.format("com.databricks.spark.csv")\
        .option("header", "true")\
        .option("inferschema", "true")\
        .option("mode", "DROPMALFORMED")\
        .load(train_business).rdd

    # Step1: find nn for users using review text
    userReviewVec, lshf, transformer, idMap, stop_words = findNN(rawData_review)

    print "Step1 Completed"

    # Step2: find business attr mappings
    busiAttrMap = findBusiAttrMapping(rawData_business)

    print "Step2 Completed"

    # Step3: get user business map
    userBusiMap = rawData_review.map(lambda x: (x[1], [x[4]])).reduceByKey(lambda x, y: x + y).collectAsMap()

    print "Step3 Completed"

    # Step4: for each user in knn find its business, then compute a weighted vote on their business
    #userReviewVec = userReviewVec.collectAsMap()
    #print(weightedVote(userReviewVec['IjVuk0tawvT0ygazmrBQEg'], userBusiMap, busiAttrMap))
    userReviewVec = userReviewVec.map(lambda x: (x[0], weightedVote(x[1], userBusiMap, busiAttrMap)))
    #print userReviewVec.collectAsMap()['IjVuk0tawvT0ygazmrBQEg']

    print "Step4 Completed"

    # Step5: find true business mapping
    # run train on train for test first
    #true_review = rawData_review.map(lambda x: (x[1], busiAttrMap[str(x[4])]))
    true_review = rawData_review.map(lambda x: (x[1], busiAttrMap[str(x[4])])).collectAsMap()

    print "Step5 Completed"

    # Step6: join prediction and true val
    #result = userReviewVec.collect()
    #result2 = true_review.collect()#.join(true_review)

    result = userReviewVec.filter(lambda x: x[0] in true_review).map(lambda x: (x[0], (x[1], true_review[x[0]])))

    print "Step6 Completed"

    # Step7: Compute error between prediction and true mapping
    MSE = result.map(lambda x: computeMSE(x[1][0], x[1][1])).collect()
    MSE = np.mean(MSE, axis=0)
    RMSE = MSE ** 0.5

    print "Step7 Completed"

    # Step8: Output the results

    print RMSE

    with open('result_train.txt', 'w') as f:
        f.writelines([str(RMSE)])

    result = result.sortByKey()
    if os.path.exists('result/train_result'):
        shutil.rmtree('result/train_result')
    result.saveAsTextFile('result/train_result')

    print "Step8 Completed"

    # Step9: Run validation data

    valid_path = 'PA/Restaurants/valid/'
    valid_review = valid_path + 'PA_valid_yelp_academic_dataset_review.csv'
    valid_business = valid_path + 'PA_valid_yelp_academic_dataset_business.csv'

    runPrediction(sc, sqlContext, valid_review, valid_business, schema_review, userBusiMap, busiAttrMap, lshf, transformer, idMap, stop_words, "valid")

    print "Step9 Completed"

    # Step10: Run Test data

    test_path = 'PA/Restaurants/test/'
    test_review = test_path + 'PA_test_yelp_academic_dataset_review.csv'
    test_business = test_path + 'PA_test_yelp_academic_dataset_business.csv'

    runPrediction(sc, sqlContext, test_review, test_business, schema_review, userBusiMap, busiAttrMap, lshf, transformer, idMap, stop_words, "test")

    print "Step10 Completed"

    return

if __name__ == '__main__':
    main()
