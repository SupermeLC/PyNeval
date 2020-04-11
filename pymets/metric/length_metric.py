from anytree import NodeMixin, iterators, RenderTree, PreOrderIter
from pymets.model.euclidean_point import EuclideanPoint,Line
from pymets.model.swc_node import SwcTree
from pymets.metric.utils.edge_match_utils import get_match_edges
from pymets.metric.utils.config_utils import get_default_threshold
from pymets.io.read_json import read_json
from pymets.io.save_swc import save_as_swc, swc_to_list
from pymets.io.read_swc import adjust_swcfile
from pymets.io.read_config import read_float_config, read_path_config
from test.test_model.length_metric.cprofile_test import do_cprofile

import time
import os, platform


def length_metric_2(gold_swc_tree=None, test_swc_tree=None,
                    rad_threshold=-1.0, len_threshold=0.2, detail_path=None, DEBUG=True):
    vertical_tree = []

    test_swc_tree.get_lca_preprocess()
    match_edges, un_match_edges = get_match_edges(gold_swc_tree, test_swc_tree,  # tree data
                                                  vertical_tree,  # a empty tree helps
                                                  rad_threshold, len_threshold, detail_path, DEBUG=False)  # configs
    # if detail_path is not None:
    #     save_as_swc(object=un_match_edges, file_path=detail_path)
    match_length = 0.0
    for line_tuple in match_edges:
        match_length += line_tuple[0].parent_distance()

    gold_total_length = round(gold_swc_tree.length(), 8)
    test_total_length = round(test_swc_tree.length(), 8)
    match_length = round(match_length, 8)

    if DEBUG:
        print("match_length a = {}, gold_total_length = {}, test_total_length = {}"
              .format(match_length, gold_total_length, test_total_length))
    return match_length/gold_total_length, match_length/test_total_length, vertical_tree


def length_metric_1(gold_swc_tree=None, test_swc_tree=None, DEBUG=False):
    gold_total_length = gold_swc_tree.length()
    test_total_length = test_swc_tree.length()

    if DEBUG:
        print("gold_total_length = {}, test_total_length = {}"
              .format(gold_total_length, test_total_length))
    return 1 - test_total_length/gold_total_length


@do_cprofile("./mkm_run.prof")
def length_metric(gold_swc_tree, test_swc_tree, abs_dir, config):
    # remove old pot mark
    gold_swc_tree.type_clear(5)
    test_swc_tree.type_clear(4)

    # get config threshold
    rad_threshold = read_float_config(config=config, config_name="rad_threshold", default=-1.0)
    len_threshold = read_float_config(config=config, config_name="len_threshold", default=0.2)

    # get config detail path
    detail_path = read_path_config(config=config, config_name="detail", abs_dir=abs_dir, default=None)

    # get config method
    if config["method"] == 1:
        ratio = length_metric_1(gold_swc_tree=gold_swc_tree,
                                test_swc_tree=test_swc_tree)
        print("1 - test_length / gold_length= {}".format(ratio))
        return ratio
    elif config["method"] == 2:
        # check every edge in test, if it is overlap with any edge in gold three
        recall, precision, vertical_tree = length_metric_2(gold_swc_tree=test_swc_tree,
                                                           test_swc_tree=gold_swc_tree,
                                                           rad_threshold=rad_threshold,
                                                           len_threshold=len_threshold,
                                                           detail_path=detail_path,
                                                           DEBUG=False)
        print("Recall = {}, Precision = {}".format(recall, precision))
        return recall, precision, vertical_tree
    else:
        raise Exception("[Error: ] Read config info method {}. length metric only have 1 and 2 two methods".format(
            config["method"]
        ))


# length metric interface connect to webmets
def web_length_metric(gold_swc, test_swc, method, rad_threshold, len_threshold):
    gold_tree = SwcTree()
    test_tree = SwcTree()

    gold_tree.load_list(adjust_swcfile(gold_swc))
    test_tree.load_list(adjust_swcfile(test_swc))

    config = {
        'method': method,
        'len_threshold': len_threshold,
        'rad_threshold': rad_threshold
    }

    recall, precision, vertical_tree = length_metric(gold_swc_tree=gold_tree,
                                                     test_swc_tree=test_tree,
                                                     abs_dir="",
                                                     config=config)
    # gold_tree.radius_limit(10)
    # test_tree.radius_limit(10)

    result = {
        'recall': recall,
        'precision': precision,
        'gold_swc': swc_to_list(gold_tree),
        'test_swc': swc_to_list(test_tree),
        'vertical_swc': vertical_tree
    }
    return result


if __name__ == "__main__":
    goldtree = SwcTree()

    testTree = SwcTree()
    goldtree.load("/home/benniehan/00_program/PyMets/test/data_example/gold/34_23_10_gold.swc")
    testTree.load("/home/benniehan/00_program/PyMets/test/data_example/test/34_23_10_test.swc")

    start = time.time()
    length_metric(gold_swc_tree=goldtree,
                  test_swc_tree=testTree,
                  abs_dir="/home/benniehan/00_program/PyMets",
                  config=read_json("/home/benniehan/00_program/PyMets/config/length_metric.json"))
    length_metric(gold_swc_tree=testTree,
                  test_swc_tree=goldtree,
                  abs_dir="/home/benniehan/00_program/PyMets",
                  config=read_json("/home/benniehan/00_program/PyMets/config/length_metric.json"))
    print("time cost = {}".format(time.time() - start))