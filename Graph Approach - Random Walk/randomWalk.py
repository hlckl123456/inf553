# coding=utf-8

import os
import sys
import pickle
import random
import numpy as np
import networkx as nx
from operator import add
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from sklearn.neighbors import NearestNeighbors
from pyspark.sql import SQLContext

from operator import itemgetter

def filterStopwords(s, stop_words):
    sb = ""
    if not s:
        return sb

    for word in s.split(" "):
        if word.lower() not in stop_words:
            sb += word + " "
    return sb[:-1]


def addEdge(edgeList, id, text, top_x_words):
    if not text:
        return ""

    tokens = text.split(" ")
    for token in tokens:
        if token in top_x_words:
            edgeList.append((id, token))
            edgeList.append((token, id))

    return ""


def toList(x1, x2, x3, x4, x5, x6):
    res = []
    res.extend(x1)
    res.extend(x2)
    res.extend([x3, x4, x5, x6])
    return res


def adjustWeight(tuples, total_topic):
    newWeights = []
    for tuple in tuples:
        newScore = tuple[1] * ((tuple[0] + 1) / total_topic)
        newWeights.append((tuple[0], newScore))

    return newWeights


# TODO: detect topic counts
def buildGraph(sc, sqlContext, LDAU, LDAB, busiAttrMap):
    total_topic = 20.0

    txt_g = 'graph.gml'
    txt_u = 'userNodes.txt'
    txt_b = 'businessNodes.txt'
    txt_ubm = 'userBusiMap.txt'

    # load review_train.csv
    reviewPath = 'PA/Restaurants/train/PA_train_yelp_academic_dataset_review.csv'
    rawData_review = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .load(reviewPath).rdd

    '''
    # find user -> business id edges
    '''

    # key (user, business) value 1/total num business
    ubDirectEdges = rawData_review.map(lambda x: (x[1], x[3])).groupByKey()\
        .flatMap(lambda x: ([(x[0], z, 1.0 / len(x[1])) for z in x[1]])).collect()

    '''
    # compute similar businesses
    '''
    business_flattened = {}
    for key, value in busiAttrMap.items():
        business_flattened[key] = toList(value[0], value[1], value[2], value[3], value[4], value[5])
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='cosine').fit(np.array(business_flattened.values()))
    nbrs_keys = np.array(business_flattened.keys())

    G = nx.DiGraph()
    # find business -> other business edge, similarity score.
    for key, value in business_flattened.items():
        distances, indices = nbrs.kneighbors([value])
        nbr_bid = nbrs_keys[indices][0]
        nbr_edges = zip(nbr_bid, distances[0])

        # cosine similarity is [0, 1], we want to cap this at 0.1, while taking 1 - cs as the edge weight
        # since cs = 0 means exactly the same
        for nbr_edge in nbr_edges:
            if key != nbr_edge[0]:
                G.add_edge(key, nbr_edge[0], value=(1 - nbr_edge[1]) / 10.0)

    '''
    Get Unique Id maps
    '''

    # find all edges
    stg1 = LDAU.map(lambda x: (x[0], adjustWeight(x[1], total_topic)))\
        .map(lambda x: (x[0], sorted(x[1], key=itemgetter(1), reverse=True)[:3]))\
        .flatMap(lambda x: map(lambda e: (x[0], e), x[1])).collect()
    stg2 = LDAB.map(lambda x: (x[0], adjustWeight(x[1], total_topic)))\
        .map(lambda x: (x[0], sorted(x[1], key=itemgetter(1), reverse=True)[:3]))\
        .flatMap(lambda x: map(lambda e: (x[0], e), x[1])).collect()

    # perform reduce on edges
    # key = (node1, node2)
    # result = (node1, node2, count) which serves as our edge result
    edges_usr = sc.parallelize(stg1).map(lambda x: (x[0], x[1][0], x[1][1])).collect()
    edges_bus = sc.parallelize(stg2).map(lambda x: (x[0], x[1][0], x[1][1])).collect()

    # save nodes and edges to gml file
    #G = nx.Graph()
    # user<->topic edges
    for edge in edges_usr:
        G.add_node(edge[0])
        G.add_node(edge[1])
        G.add_edge(edge[0], edge[1], value=edge[2])
        G.add_edge(edge[1], edge[0], value=edge[2] / 2)  # halves the weight from review to user

    # business<->topic edges
    for edge in edges_bus:
        G.add_node(edge[0])
        G.add_node(edge[1])
        G.add_edge(edge[0], edge[1], value=edge[2] / 2)  # halves the weight from business to review
        G.add_edge(edge[1], edge[0], value=edge[2])

    # use->business edges
    for tp in ubDirectEdges:
        G.add_edge(tp[0], tp[1], value=tp[2])

    nx.write_gml(G, txt_g)

    u_id = LDAU.map(lambda x: x[0]).collect()
    b_id = LDAB.map(lambda x: x[0]).collect()

    # save id sets to txt files
    with open(txt_u, 'wb') as f:
        pickle.dump(u_id, f)

    with open(txt_b, 'wb') as f:
        pickle.dump(b_id, f)

    return G, b_id, u_id


def collapse(tp1, tp2):
    arr = [0] * len(tp1)
    for i in range(len(tp1)):
        arr[i] = max(tp1[i], 0) + max(tp2[i], 0)

    return arr


def stripAndDivide(count, arr):
    new_arr = [0] * len(arr)
    for i in range(len(arr)):
        new_arr[i] = (i, max(arr[i], 0) / count)

    return new_arr


def loadLDA(sqlContext, lda_upath, lda_bpath):
    ldau = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .load(lda_upath).rdd
    ldab = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .load(lda_bpath).rdd

    ldau = ldau.map(lambda x: (x[0], (1, x[1:]))).reduceByKey(lambda x, y: (x[0] + y[0], collapse(x[1], y[1])))\
        .map(lambda x: (x[0], stripAndDivide(x[1][0], x[1][1])))
    ldab = ldab.map(lambda x: (x[0], (1, x[1:]))).reduceByKey(lambda x, y: (x[0] + y[0], collapse(x[1], y[1])))\
        .map(lambda x: (x[0], stripAndDivide(x[1][0], x[1][1])))

    return ldau, ldab


def findNeighbors(G, node, neighbors):
    n = []
    p = []
    for neighbor in neighbors:
        n.append(neighbor)
        p.append(G.get_edge_data(node, neighbor)['value'])

    p = np.array(p) / np.sum(p)

    return np.random.choice(n, 1, False, p)[0]


def graphWalk(G, B, U, type, sqlContext):
    rdmSeed = 104729
    np.random.seed(rdmSeed)

    # load review_train.csv
    reviewPath = 'PA/Restaurants/' + type + '/PA_' + type + '_yelp_academic_dataset_review.csv'
    rawData_review = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .load(reviewPath).rdd

    userBusiMap = rawData_review.map(lambda x: (x[1], [x[3]])).reduceByKey(lambda x, y: x + y).collectAsMap()

    # Dictionary that associate nodes with the amount of times it was visited
    visitedVertices = {}

    for node in U:
        # Choose a random start node
        vertexid = node

        # Execute the random walk with size 10000 (10000 steps)
        cnt = 0
        while cnt < 5000:
            # assign vertexid = start with probability p and vertexid with probability 1 - p
            alpha = random.random()
            if alpha < 0.05:
                vertexid = node

            # Visualize the vertex neighborhood
            Vertex_Neighbors = G.neighbors(vertexid)

            # Choose a vertex from the vertex neighborhood to start the next random walk
            vertexid = findNeighbors(G, vertexid, list(Vertex_Neighbors))
            # Iteration counter increment

            cnt += 1

            # Accumulate the amount of times each vertex is visited
            if vertexid not in B:
                continue

            if node in visitedVertices and vertexid in visitedVertices[node]:
                visitedVertices[node][vertexid] += 1
            elif node in visitedVertices:
                visitedVertices[node][vertexid] = 1
            else:
                visitedVertices[node] = {}
                visitedVertices[node][vertexid] = 1

    # Organize the vertex list in most visited decrescent order
    for k, v in visitedVertices.items():
        visitedVertices[k] = sorted(v.iteritems(), key=lambda (k,v): (v,k), reverse=True)[10:60]

    # compute hit ratio
    hits = []
    for k, v in visitedVertices.items():
        found = False
        v_bid, v_visits = zip(*v)
        for bid in v_bid:
            if k in userBusiMap and bid in userBusiMap[k]:
                found = True
                hits.append(1)
                break
        if not found:
            hits.append(0)

    # save the hit ratio to txt
    if len(hits) == 0: hit_ratios = 0
    else: hit_ratios = np.mean(hits)
    with open('random_walk_hit_ratio.txt', 'w') as f:
        f.writelines([str(hit_ratios)])

    return visitedVertices


def getIdMaps(sqlContext, path, B, U):
    rawData_review = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .load(path).rdd

    uid = rawData_review.map(lambda x: (x[1], 1)).reduceByKey(add).map(lambda x: x[0]).filter(lambda x: x in U).collect()
    bid = rawData_review.map(lambda x: (x[3], 1)).reduceByKey(add).map(lambda x: x[0]).filter(lambda x: x in B).collect()

    return bid, uid


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
        return 1.0


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
        return 2.0


def fillPrange(prange):
    if prange == None:
        return 1.0
    else:
        return prange


def findBusiAttrMapping(rawData_business):
    data_business = rawData_business.map(lambda x: (x[82], ([x[4], x[20], x[22], x[25], x[60], x[73]]),
                                                    ([x[45], x[46], x[47], x[48], x[49], x[50]]), x[99], x[3], x[68], x[75]))

    data_business = data_business.map(lambda x: (str(x[0]), (toVec(x[1]), toVec(x[2]), x[3], alcholToScaler(x[4]), noiseLevelToScaler(x[5]), fillPrange(x[6]))))

    return data_business.collectAsMap()


def main():
    reload(sys)
    sys.setdefaultencoding("utf-8")

    # spark config
    conf = SparkConf()
    conf.setMaster("local").setAppName("MemoryBasedCF")
    conf.set("spark.network.timeout", "3600s")
    conf.set("spark.executor.heartbeatInterval", "3000s")
    conf.set("spark.executor.memory", "10g")
    conf.set("spark.driver.memory", "4g")

    sc = SparkContext(conf=conf)
    sc.setCheckpointDir("checkpoint")
    sqlContext = SQLContext(sc)

    graph_path = 'graph.gml'

    run = sys.argv[1]

    frdnWalk_train = "randomWalkResult_train.txt"
    frdnWalk_valid = "randomWalkResult_valid.txt"
    frdnWalk_test = "randomWalkResult_test.txt"

    # find business attr mappings
    train_business = 'PA/Restaurants/train/PA_train_yelp_academic_dataset_business.csv'
    rawData_business = sqlContext.read.format("com.databricks.spark.csv")\
        .option("header", "true")\
        .option("inferschema", "true")\
        .option("mode", "DROPMALFORMED")\
        .load(train_business).rdd

    busiAttrMap = findBusiAttrMapping(rawData_business)

    if os.path.exists(graph_path):
        G = nx.read_gml(graph_path)
        with open('businessNodes.txt', 'rb') as f:
            B = pickle.load(f)
        with open('userNodes.txt', 'rb') as f:
            U = pickle.load(f)

    else:
        lda_upath = "user_reviews_topic.csv"
        lda_bpath = "business_reviews_topic.csv"
        LDAU, LDAB = loadLDA(sqlContext, lda_upath, lda_bpath)
        G, B, U = buildGraph(sc, sqlContext, LDAU, LDAB, busiAttrMap)

    print("Graph Loaded")

    if run == 'R':
        print('Walk Start')
        rdnWalkRes = graphWalk(G, B, U, 'train', sqlContext)
        with open(frdnWalk_train, 'wb') as f:
            pickle.dump(rdnWalkRes, f)
    elif run == 'V':
        B, U = getIdMaps(sqlContext, 'PA/Restaurants/valid/PA_valid_yelp_academic_dataset_review.csv', B, U)
        print('Walk Start')
        rdnWalkRes = graphWalk(G, B, U, 'valid', sqlContext)
        with open(frdnWalk_valid, 'wb') as f:
            pickle.dump(rdnWalkRes, f)
    elif run == 'T':
        B, U = getIdMaps(sqlContext, 'PA/Restaurants/test/PA_test_yelp_academic_dataset_review.csv', B, U)
        print('Walk Start')
        rdnWalkRes = graphWalk(G, B, U, 'test', sqlContext)
        with open(frdnWalk_test, 'wb') as f:
            pickle.dump(rdnWalkRes, f)

    return

if __name__ == '__main__':
    main()
