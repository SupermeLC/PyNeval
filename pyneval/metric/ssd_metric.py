# -*- coding: utf-8 -*-
"""a geometry metric method, SSD metric

This module implement a geometry metric method called Substantial Spatial Distance.
relative paper: https://doi.org/10.1038/nbt.1612
title, auther

Example:
    In command line:
    $ pyneval --gold ./data\\test_data\\ssd_data\\gold\\c.swc
              --test .\\data\\test_data\\ssd_data\\test\\c.swc
              --metric ssd_metric
Attributes:
    None

Todos:
    None
"""
import sys

import jsonschema

from pyneval.model import swc_node
from pyneval.tools import re_sample
from pyneval.metric.utils import point_match_utils

from pyneval.metric.utils.metric_manager import get_metric_manager

metric_manager = get_metric_manager()


class SsdMetric(object):
    """
    ssd metric
    """

    def __init__(self, config):
        self.debug = config["debug"] if config.get("debug") is not None else False
        self.threshold_mode = config["threshold_mode"]
        self.ssd_threshold = config["ssd_threshold"]
        self.up_sample_threshold = config["up_sample_threshold"]
        self.scale = config["scale"]

    def get_mse(self, src_tree, tar_tree, ssd_threshold=2.0, mode=1):
        """ calculate the minimum square error of two trees
        find the closest node on the tar_tree for each node on the src tree, calculate the average
        distance of these node pairs.

        Args:
            src_tree(SwcTree):
            tar_tree(SwcTree):
            ssd_threshold(float): distance will be count into the res
                only if two nodes' distance is larger than this threshold.
            mode(1 or 2):
                1 means static threshold, equal to ssd_threshold.
                2 means dynamic threshold, equal to ssd_threshold * src_node.threshold

        Returns:
            dis(float): average minimum distance of node, distance must be larger than ssd_threshold
            num(float): The number of pairs of nodes that are counted when calculating distance

        Raise:
            None
        """
        dis, num = 0, 0
        kdtree, pos_node_dict = point_match_utils.create_kdtree(tar_tree.get_node_list())
        for node in src_tree.get_node_list():
            if node.is_virtual():
                continue
            target_pos = kdtree.search_knn(list(node.get_center_as_tuple()), k=1)[0]
            target_node = pos_node_dict[tuple(target_pos[0].data)]

            cur_dis = target_node.distance(node)

            if mode == 1:
                threshold = ssd_threshold
            else:
                threshold = ssd_threshold * node.radius()

            if cur_dis >= threshold:
                node._type = 9
                dis += cur_dis
                num += 1
        try:
            dis /= num
        except ZeroDivisionError:
            dis = num = 0
        return dis, num

    def run(self, gold_swc_tree, test_swc_tree):
        gold_swc_tree.rescale(self.scale)
        test_swc_tree.rescale(self.scale)
        u_gold_swc_tree = re_sample.up_sample_swc_tree(swc_tree=gold_swc_tree,
                                                       length_threshold=self.up_sample_threshold)
        u_test_swc_tree = re_sample.up_sample_swc_tree(swc_tree=test_swc_tree,
                                                       length_threshold=self.up_sample_threshold)
        u_gold_swc_tree.set_node_type_by_topo(root_id=1)
        u_test_swc_tree.set_node_type_by_topo(root_id=5)

        g2t_score, g2t_num = self.get_mse(src_tree=u_gold_swc_tree, tar_tree=u_test_swc_tree,
                                     ssd_threshold=self.ssd_threshold, mode=self.threshold_mode)
        t2g_score, t2g_num = self.get_mse(src_tree=u_test_swc_tree, tar_tree=u_gold_swc_tree,
                                     ssd_threshold=self.ssd_threshold, mode=self.threshold_mode)

        if self.debug:
            print("recall_num = {}, pre_num = {}, gold_tot_num = {}, test_tot_num = {} {} {}".format(
                g2t_num, t2g_num, u_gold_swc_tree.size(), u_test_swc_tree.size(), gold_swc_tree.length(),
                test_swc_tree.length()
            ))

        res = {
            "avg_score": (g2t_score + t2g_score) / 2,
            "recall": 1 - g2t_num / u_gold_swc_tree.size(),
            "precision": 1 - t2g_num / u_test_swc_tree.size()
        }

        return res, u_gold_swc_tree, u_test_swc_tree


@metric_manager.register(
    name="ssd",
    config="ssd_metric.json",
    desc="minimum square error between up-sampled gold and test trees",
    alias=['SM'],
    public=True,
)
def ssd_metric(gold_swc_tree: swc_node.SwcTree, test_swc_tree: swc_node.SwcTree, config: dict):
    """Main function of SSD metric.
    Args:
        gold_swc_tree(SwcTree):
        test_swc_tree(SwcTree):
        config(Dict):
            The keys of 'config' is the name of configs, and the items are config values
    Example:
        test_tree = swc_node.SwcTree()
        gold_tree = swc_node.SwcTree()
        gold_tree.load("..\\..\\data\\test_data\\ssd_data\\gold\\c.swc")
        test_tree.load("..\\..\\data\\test_data\\ssd_data\\test\\c.swc")
        score, recall, precision = ssd_metric(gold_swc_tree=gold_tree,
                                              test_swc_tree=test_tree,
                                              config=config)
    Returns:
        tuple: contain three values to demonstrate metric result
            avg_score(float): average distance of nodes between gold and test swc trees.
            precision(float): percentage of nodes that are matched compared to test tree
            recall(float): percentage of nodes that are matched compared to gold tree

    Raises:
        None
    """
    ssd = SsdMetric(config)
    return ssd.run(gold_swc_tree, test_swc_tree)

# if __name__ == "__main__":
#     test_tree = swc_node.SwcTree()
#     gold_tree = swc_node.SwcTree()
#
#     sys.setrecursionlimit(10000000)
#     gold_tree.load("../../data/test_data/geo_metric_data/gold_fake_data5.swc")
#     test_tree.load("../../data/test_data/geo_metric_data/test_fake_data5.swc")
#
#     from pyneval.metric.utils import config_utils
#     config = config_utils.get_default_configs("ssd")
#     config_schema = config_utils.get_config_schema("ssd")
#
#     try:
#         jsonschema.validate(config, config_schema)
#     except Exception as e:
#         raise Exception("[Error: ]Error in analyzing config json file")
#     config["detail_path"] = "..//..//output//ssd_output//ssd_detail.swc"
#
#     ssd_res,_,_ = ssd_metric(gold_swc_tree=gold_tree,
#                          test_swc_tree=test_tree,
#                          config=config)
#     print(2*ssd_res["recall"]*ssd_res["precision"])
#     print(ssd_res["recall"]+ssd_res["precision"])
#     print("ssd score = {}\n"
#           "recall    = {}%\n"
#           "precision = {}%\n"
#           "f1        = {}".
#           format(round(ssd_res["avg_score"], 2),
#                  round(ssd_res["recall"]*100, 2),
#                  round(ssd_res["precision"]*100, 2),
#                  round((2*ssd_res["recall"]*ssd_res["precision"])/(ssd_res["recall"]+ssd_res["precision"]), 2)))

if __name__ == "__main__":
    test_tree = swc_node.SwcTree()
    gold_tree = swc_node.SwcTree()
#['real', 'm_cyc', 'm_cyc_mp', 'm_mp', 'm_sim', 'm_sim_mp']
    model_name = ['real', 'm_cyc_mp', 'm_mp', 'm_sim_mp']
    # org_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
    org_rate = [0, 1, 50, 100, 150, 250]
    image_name = ['6656_2304_22016', '6656_2304_21504', '34_23_10', '2_img', '3_img', '5_img']

    sys.setrecursionlimit(10000000)
    root_dir = 'E:/Projects/Brain Tracing/unet/data/model/simGAN/exp_3'
    image_name_temp = '5_img'

    gold_swc_dir = root_dir + '/gold/' + image_name_temp + '.gold.swc'
    # goldTree.load("..\\..\\data\\lc\\sch\\gold\\Image4.swc")
    gold_tree.load(gold_swc_dir)

    from pyneval.metric.utils import config_utils

    config = config_utils.get_default_configs("ssd")
    config_schema = config_utils.get_config_schema("ssd")

    try:
        jsonschema.validate(config, config_schema)
    except Exception as e:
        raise Exception("[Error: ]Error in analyzing config json file")
    config["detail_path"] = "..//..//output//ssd_output//ssd_detail.swc"



    test_swc_dir = root_dir + '/result_adj/origin/' + image_name_temp + '.swc'
    test_tree.load(test_swc_dir)
    ssd_res, _, _ = ssd_metric(gold_swc_tree=gold_tree,test_swc_tree=test_tree,config=config)

    f1 = 2 * ssd_res["recall"] * ssd_res["precision"] / (ssd_res["recall"] + ssd_res["precision"] + 0.000001)
    print("ssd score = {:5} precision = {:5} recall = {:5} f1_score = {}".format(round(ssd_res["avg_score"], 2), round(ssd_res["precision"], 3), round(ssd_res["recall"], 3), round(f1, 3)))

    pause


    for model_name_temp in model_name:
        for org_rate_temp in org_rate:

            testTree = swc_node.SwcTree()
            # test_swc_dir = root_dir + '/result_adj/origin/' + image_name_temp + '.swc'
            test_swc_dir = root_dir + '/result_old/' + model_name_temp + '/' + image_name_temp + '.' + str(org_rate_temp) + '.swc'
            testTree.load(test_swc_dir)

            ssd_res, _, _ = ssd_metric(gold_swc_tree=gold_tree,test_swc_tree=testTree,config=config)

            f1 = 2 * ssd_res["recall"] * ssd_res["precision"] / (ssd_res["recall"] + ssd_res["precision"] + 0.000001)

            print("model = {:10} rate = {:5} ssd score = {:5} precision = {:5} recall = {:5} f1_score = {}".format(model_name_temp, org_rate_temp,round(ssd_res["avg_score"], 2), round(ssd_res["precision"], 3), round(ssd_res["recall"], 3), round(f1, 3)))

            # print("ssd score = {}\n"
            #       "precision = {}%\n"
            #       "recall = {}%\n"
            #       "f1_score = {}%".
            #       format(round(ssd_res["avg_score"], 2),
            #              round(ssd_res["precision"], 3),
            #              round(ssd_res["recall"], 3),
            #              round(f1, 3)))
        print("============================================================================================")