import sys
import jsonschema

from pyneval.model import swc_node
from pyneval.metric.utils import edge_match_utils
from pyneval.io import read_swc
from pyneval.io import swc_writer
from pyneval.metric.utils.metric_manager import get_metric_manager

metric_manager = get_metric_manager()


class LengthMetric(object):
    """
    length metric
    """

    def __init__(self, config):
        # read config
        self.rad_mode = config["rad_mode"]
        self.rad_threshold = config["rad_threshold"]
        self.len_threshold = config["len_threshold"]
        self.scale = config["scale"]
        self.debug = config["debug"]
        self.detail_path = config.get("detail_path")

    def length_metric_run(self, gold_swc_tree=None, test_swc_tree=None,
                          rad_threshold=-1.0, len_threshold=0.2):
        """
        get matched edge set and calculate recall and precision
        Args:
            gold_swc_tree(SwcTree):
            test_swc_tree(SwcTree):
            rad_threshold(float): threshold of key point radius
            len_threshold(float): threshold of length of the matching edges
            debug(bool): list debug info ot not
        Returns:
            tuple: contain two values to demonstrate metric result
                precision(float): percentage of total length of edges that are matched compared to test tree
                recall(float): percentage of total length of edges that are matched compared to gold tree
        Raises:
            None
        """
        # get matched edge set
        match_edges, test_match_length = edge_match_utils.get_match_edges(gold_swc_tree=gold_swc_tree,
                                                                          test_swc_tree=test_swc_tree,
                                                                          rad_threshold=rad_threshold,
                                                                          len_threshold=len_threshold,
                                                                          debug=self.debug)
        # calculate the sum of matched length and total length of gold and test tree
        match_length = 0.0
        for line_tuple in match_edges:
            match_length += line_tuple[0].parent_distance()

        gold_total_length = round(gold_swc_tree.length(), 8)
        test_total_length = round(test_swc_tree.length(), 8)
        match_length = round(match_length, 8)
        test_match_length = round(test_match_length, 8)

        if self.debug:
            print("match_length = {}, test_match_length = {}, gold_total_length = {}, test_total_length = {}"
                  .format(match_length, test_match_length, gold_total_length, test_total_length))
        # calculate recall and precision
        if gold_total_length != 0:
            recall = round(match_length / gold_total_length, 8)
        else:
            recall = 0

        if test_total_length != 0:
            precision = round(test_match_length / test_total_length, 8)
        else:
            precision = 0

        return min(recall, 1.0), min(precision, 1.0)

    def run(self, gold_swc_tree, test_swc_tree):
        """Main function of length metric.
            unpack config and run the matching function
            Args:
                gold_swc_tree(SwcTree):
                test_swc_tree(SwcTree):
            Example:
                test_tree = swc_node.SwcTree()
                gold_tree = swc_node.SwcTree()
                gold_tree.load("..\\..\\data\\test_data\\geo_metric_data\\gold_fake_data1.swc")
                test_tree.load("..\\..\\data\\test_data\\geo_metric_data\\test_fake_data1.swc")
                lm_res = length_metric(gold_swc_tree=gold_tree,
                                       test_swc_tree=test_tree,
                                       config=config)
            Returns:
                tuple: contain two values to demonstrate metric result
                    precision(float): percentage of total length of edges that are matched compared to test tree
                    recall(float): percentage of total length of edges that are matched compared to gold tree
            Raises:
                None
            """
        gold_swc_tree.rescale(self.scale)
        test_swc_tree.rescale(self.scale)
        gold_swc_tree.set_node_type_by_topo(root_id=1)
        test_swc_tree.set_node_type_by_topo(root_id=5)

        if self.rad_mode == 1:
            self.rad_threshold *= -1
        # check every edge in test, if it is overlap with any edge in gold three
        recall, precision = self.length_metric_run(gold_swc_tree=gold_swc_tree,
                                                   test_swc_tree=test_swc_tree,
                                                   rad_threshold=self.rad_threshold,
                                                   len_threshold=self.len_threshold,
                                                   )
        if self.detail_path:
            swc_writer.swc_save(gold_swc_tree, config["detail_path"][:-4] + "_gold.swc")
            swc_writer.swc_save(test_swc_tree, config["detail_path"][:-4] + "_test.swc")
        if self.debug:
            print("Recall = {}, Precision = {}".format(recall, precision))

        res = {
            "recall": recall,
            "precision": precision
        }
        return res, gold_swc_tree, test_swc_tree


# @do_cprofile("./mkm_run.prof")
@metric_manager.register(
    name="length",
    config="length_metric.json",
    desc="length of matched branches and fibers",
    public=True,
    alias=['ML']
)
def length_metric(gold_swc_tree, test_swc_tree, config):
    """Main function of length metric.
    unpack config and run the matching function
    Args:
        gold_swc_tree(SwcTree):
        test_swc_tree(SwcTree):
        config(Dict):
            keys: the name of configs
            items: config values
    Example:
        test_tree = swc_node.SwcTree()
        gold_tree = swc_node.SwcTree()
        gold_tree.load("..\\..\\data\\test_data\\geo_metric_data\\gold_fake_data1.swc")
        test_tree.load("..\\..\\data\\test_data\\geo_metric_data\\test_fake_data1.swc")
        lm_res = length_metric(gold_swc_tree=gold_tree,
                               test_swc_tree=test_tree,
                               config=config)
    Returns:
        tuple: contain two values to demonstrate metric result
            precision(float): percentage of total length of edges that are matched compared to test tree
            recall(float): percentage of total length of edges that are matched compared to gold tree
    Raises:
        None
    """

    length_metric = LengthMetric(config)
    return length_metric.run(gold_swc_tree, test_swc_tree)


if __name__ == "__main__":
    goldTree = swc_node.SwcTree()
    testTree = swc_node.SwcTree()
    sys.setrecursionlimit(10000000)

    model_name = ['real', 'm_sim', 'm_sim_mp', 'm_cyc', 'm_cyc_mp', 'm_mp']
    org_rate = [0, 1, 50, 100, 150, 250]
    image_name = ['6656_2304_22016', '6656_2304_21504', '34_23_10', '2_img', '3_img', '5_img']

    root_dir = 'E:/Projects/Brain Tracing/unet/data/model/simGAN/exp_3'
    image_name_temp = '6656_2304_22016'

    gold_swc_dir = root_dir + '/gold/' + image_name_temp + '.gold.swc'
    # goldTree.load("..\\..\\data\\lc\\sch\\gold\\Image4.swc")
    goldTree.load(gold_swc_dir)


    from pyneval.metric.utils import config_utils

    config = config_utils.get_default_configs("length")
    config_schema = config_utils.get_config_schema("length")
    try:
        jsonschema.validate(config, config_schema)
    except Exception as e:
        raise Exception("[Error: ]Error in analyzing config json file")
    config["detail_path"] = "..\\..\\output\\length_output\\length_metric_detail.swc"

    # test_swc_dir = root_dir + '/result_adj/origin/' + image_name_temp + '.swc'
    # testTree.load(test_swc_dir)
    # lm_res, _, _ = length_metric(gold_swc_tree=testTree,
    #                              test_swc_tree=goldTree,
    #                              config=config)
    #
    # f1 = 2 * lm_res["recall"] * lm_res["precision"] / (lm_res["recall"] + lm_res["precision"] + 0.000001)
    # print("precision = {}\n"
    #       "recall = {}\n"
    #       "f1_score = {}%".format(lm_res["precision"], lm_res["recall"], round(f1, 3)))
    # pause



    for model_name_temp in model_name:
        for org_rate_temp in org_rate:
            testTree = swc_node.SwcTree()
            # test_swc_dir = root_dir + '/result_adj/origin/' + image_name_temp + '.swc'
            test_swc_dir = root_dir + '/result_old/' + model_name_temp + '/' + image_name_temp + '.' + str(org_rate_temp) + '.swc'
            testTree.load(test_swc_dir)

            lm_res,_,_ = length_metric(gold_swc_tree=goldTree,
                                   test_swc_tree=testTree,
                                   config=config)

            f1 = 2 * lm_res["recall"] * lm_res["precision"] / (lm_res["recall"] + lm_res["precision"] + 0.000001)

            print("model = {:10} rate = {} precision = {:5} recall = {:5} f1_score = {}".format(model_name_temp, org_rate_temp, round(lm_res["precision"],3) ,round(lm_res["recall"],3), round(f1, 3)))
            # print("precision = {}\n"
            #       "recall = {}\n"
            #       "f1_score = {}%".format(lm_res["precision"], lm_res["recall"], round(f1, 3)))
        print("========================================================================")
    #

    # lm_res, _, _ = length_metric(gold_swc_tree=testTree,
    #                              test_swc_tree=goldTree,
    #                              config=config)
    #
    # print("recall    = {}\n"
    #       "precision = {}\n"
    #       "f1        = {}".format(lm_res["recall"], lm_res["precision"], (
    #             lm_res["recall"] * lm_res["precision"] * 2 / (lm_res["recall"] + lm_res["precision"]))))
