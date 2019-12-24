import argparse
import sys,os
from pymets.io.read_swc import read_swc_trees
from pymets.io.read_json import read_json
from pymets.metric.diadem_metric import diadem_metric
from pymets.metric.length_metric import length_metric

def read_parameters():
    parser = argparse.ArgumentParser(
        description="pymet 1.0"
    )

    parser.add_argument(
        "--test",
        "-T",
        help="the route of the test file",
        required=True,
        nargs='*',
    )
    parser.add_argument(
        "--gold",
        "-G",
        help="the route of the gold file",
        required=True
    )
    parser.add_argument(
        "--metric",
        "-M",
        help="choose a metric method",
        required=True
    )
    parser.add_argument(
        "--output",
        "-O",
        help="the route of the output file.\nif not specified, output to screen",
        required=False
    )
    parser.add_argument(
        "--config",
        "-C",
        help="special config for different metric method",
        required=False
    )
    return parser.parse_args()

def pymets(DEBUG=True):
    abs_dir = os.path.abspath("")

    sys.path.append(abs_dir)
    sys.path.append(os.path.join(abs_dir,"src"))
    sys.path.append(os.path.join(abs_dir,"test"))

    args = read_parameters()

    test_swc_files = [os.path.join(abs_dir, path) for path in args.test]
    gold_swc_file = os.path.join(abs_dir, args.gold)

    print(test_swc_files)
    print(gold_swc_file)
    metric  = args.metric
    output_dest = args.output
    config = args.config
    if config is None:
        if metric == "diadem_metric" or metric == "DM":
            config = os.path.join(abs_dir, "config\\diadem_metric.json")
        if metric == "length_metric" or metric == "LM":
            config = os.path.join(abs_dir, "config\\length_metric.json")

    if DEBUG:
        print("Config = {}".format(config))

    test_swc_trees = []
    for test_swc_file in test_swc_files:
        test_swc_trees += read_swc_trees(test_swc_file)
    gold_swc_trees = read_swc_trees(gold_swc_file)
    config = read_json(config)

    print("There are {} test image(s) and {} gold image(s)".format(len(test_swc_trees), len(gold_swc_trees)))
    if len(gold_swc_trees) == 0:
        raise Exception("[Error:  ] No gold image detected")
    if len(gold_swc_trees) > 1:
        print("[Warning:  ] More than one gold image detected, only the first one will be used")

    gold_swc_treeroot = gold_swc_trees[0]
    for test_swc_treeroot in test_swc_trees:
        if metric  == "diadem_metric" or metric  == "DM":
            diadem_metric(test_swc_treeroot, gold_swc_treeroot)
        if metric  == "length_metric" or metric  == "LM":
            result = length_metric(gold_swc_treeroot, test_swc_treeroot, config)
            print(result)

if __name__ == "__main__":
    pymets()
# python ./pymets.py --test D:\gitProject\mine\PyMets\test\data_example\test\30_18_10_test.swc --gold D:\gitProject\mine\PyMets\test\data_example\gold\30_18_10_gold.swc --metric length_metric --config D:\gitProject\mine\PyMets\test\length_metric.json
# python ./pymets.py --test D:\gitProject\mine\PyMets\test\data_example\test\30_18_10_test.swc --gold D:\gitProject\mine\PyMets\test\data_example\gold\30_18_10_gold.swc --metric length_metric
