from Tree import Tree
# General utilities
import os
import sys
import time
import signal
import datetime
import logging
import argparse
import pickle
import string
import numpy as np
import json
import csv

import random

# RP3 specific packages
from move import Move  # only necessary for tracking class below
from rdkit import Chem  # only necessary for tracking below
from compound import Compound, unpickle, CompoundDefinitionException, ChemConversionError
from chemical_compounds_state import ChemicalCompoundState
from MCTS_node import MCTS_node
from representation import Test_representation, Test_to_file
from UCT_policies import Biochemical_UCT_1, Nature_UCT, Classical_UCT_RAVE, Classical_UCT_with_bias, Classical_UCT, \
    Biochemical_UCT_1_with_RAVE, Biochemical_UCT_with_progressive_bias, Chemical_UCT_1, Biological_UCT_1, \
    Biochemical_UCT_with_toxicity
from Rollout_policies import Rollout_policy_first, Rollout_policy_random_uniform_on_biochemical_multiplication_score, Rollout_policy_random_uniform
from rewarding import Basic_Rollout_Reward, RolloutRewards
from rule_sets_examples import applicable_rules_mixed_dict, applicable_rules_10_dict
from rule_sets_similarity import get_rules_and_score, full_rules_forward_H, full_rules_retro_H, full_rules_forward_no_H, \
    full_rules_retro_no_H
from pathway import Pathway
from pathway_scoring import RandomPathwayScorer, constant_pathway_scoring, null_pathway_scoring, \
    biological_pathway_scoring, biochemical_pathway_scoring
from tree_viewer import Tree_viewer
from organisms import detectable_cmpds_H, ecoli_chassis_H, Test_organism_H, iJO1366_chassis_H, core_ecoli_H, core_ecoli_noH, bsubtilis_H, bsubtilis_noH
from organisms import detectable_cmpds_noH, ecoli_chassis_noH, Test_organism_noH, import_organism_from_csv, iJO1366_chassis_noH
# General Configuration
from config import *
if use_toxicity:
    from compound_scoring import toxicity_scorer

class InvalidSink(Exception):
    """Class for raising exception if sink is invalid."""

    def __init__(self):
        self.reason = "InvalidSink"
        self.message = "Invalid (empty) sink"

class CompoundInSink(Exception):
    """Class for raising exception if compound already in sink."""

    def __init__(self, folder_to_save, name):
        self.message = "Compound {} already in organism".format(name)
        self.reason = "CompoundInSink"
        with open("{}/in_sink".format(folder_to_save), "w") as results_file:
            pass

class RunModeError(Exception):
    """Class for Tree Mode exception."""

    def __init__(self, retrosynthesis, biosensor):
        self.message = "Choose between retrosynthesis ({}) and biosensor ({})".format(retrosynthesis, biosensor)
        self.reason = "RunModeError"

class IncorrectTreeLoading(Exception):
    """Class for Conflicts between trees when loading a tree for search extension."""

    def __init__(self, message):
        self.message = message
        self.reason = "IncorrectTreeLoading"
        

if __name__ == '__main__':
    
    def define_folder_to_save(folder):
        if folder is None:
            folder_to_save = os.path.join('debugging_results', args.c_name)
        else:
            folder_to_save = folder
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save, exist_ok=True)
        if not os.path.exists(os.path.join(folder_to_save, 'pickles')):
            os.mkdir(os.path.join(folder_to_save, 'pickles'))
        return folder_to_save

    def get_representation(representation):
        if representation:
            representation = Test_representation
        else:
            representation = Test_to_file
        return representation
    
    def get_progressive_bias(progressive_bias_strategy):
        try:
            progressive_bias_strategy = int(progressive_bias_strategy)
        except ValueError:
            progressive_bias_strategy = progressive_bias_strategy
            if progressive_bias_strategy == "None":
                progressive_bias_strategy = None
        return progressive_bias_strategy

    def get_organism(biosensor, organism_name = "none", complementary_sink=None, add_Hs=True):
        """
        Imports sinks.
        - detectable compounds for biosensors
        - all avaialble sinks.
        - complementary sink for media supplementation
        """
        if biosensor:
            logging.info("Using detectable compounds as sink as running in biosensor mode.")
            if add_Hs:
                organism = detectable_cmpds_H
            else:
                detectable_cmpds_noH
        else:
            if organism_name == "none":
                if complementary_sink is None:
                    logging.warning("Need to specify a sink")
                    raise InvalidSink
                else:
                    organism = import_organism_from_csv(complementary_sink, add_Hs=add_Hs)
                    logging.info("File {} is the sink".format(complementary_sink))
            elif organism_name == "test":
                if add_Hs:
                    organism = Test_organism_H
                else:
                    organism = Test_organism_noH
            elif organism_name == "ecoli":
                if add_Hs:
                    organism = ecoli_chassis_H
                else:
                    organism = ecoli_chassis_noH
            elif organism_name == "core_ecoli":
                if add_Hs:
                    organism = core_ecoli_H
                else:
                    organism = core_ecoli_noH
            elif organism_name == "ijo1366":
                if add_Hs:
                    organism = iJO1366_chassis_H
                else:
                    organism = iJO1366_chassis_noH
            elif organism_name == "bsubtilis":
                if add_Hs:
                    organism = bsubtilis_H
                else:
                    organism = bsubtilis_noH
            else:
                logging.warning("This organism is not implemented yet: {}".format(organism_name))
                raise NotImplementedError
        if not complementary_sink is None and organism_name != "none":
            cmpds_to_add = import_organism_from_csv(complementary_sink, add_Hs=add_Hs)
            organism.merge_states(cmpds_to_add)
            logging.info("Add compounds from {} to the sink".format(complementary_sink))
        return(organism)
    
    d = "All arguments to run a nice MCTS"
    parser = argparse.ArgumentParser(description=d)
    # Logs and saving information
    parser.add_argument("--verbose", help="Default logger is INFO, switch to DEBUG is specified",
                        dest='verbose', action='store_true', default=False)
    parser.add_argument("--log_file", help="Default logger is stderr, switch to log_file if specified",
                        default=None)
    parser.add_argument("--folder_to_save",
                        help="Folder to store results. Default: temp",
                        default="temp")
    parser.add_argument("--heavy_saving",
                        help='If True, will save the tree each max_iteration/10',
                        default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--stop_at_first_result",
                        help='If True, will stop the first time it encounters a fully solved pathway',
                        default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    # Compound information
    parser.add_argument("--c_name", help="Compound name. Defaults to None (InchiKey)",
                        default="lycosantalonol")
    # One of the next 2 arguments has to be specified to give a target structure.
    parser.add_argument("--c_smiles", help="Compound smiles", default = "CC(=CCCC(C)(C(=O)CCC1(C2CC3C1(C3C2)C)C)O)C")
    parser.add_argument("--c_inchi", help="Compound inchi", default = None)
    # Timeouts on rdkit processes
    parser.add_argument("--fire_timeout", help = "Time allowed for firing one rule on one substrate",
                        type = float, default = 1)
    parser.add_argument("--standardisation_timeout", help = "Time allowed for standardising results from one application",
                        type = float, default = 5)
    # Complementary sink: if we need to supplement the media or use another sink than the provided ones.
    parser.add_argument("--organism_name", default = "none",
                        choices = ["none", "test", "ecoli", "core_ecoli", "bsubtilis", "ijo1366"])
    parser.add_argument("--complementary_sink",
                        help="address of a csv file containing compounds to add",
                        default='/home/lmartins/RetroPathRL_LuciEdition/building_blocks.csv')
    # Visualisation
    parser.add_argument("--representation",
                        help="If activated, uses colors for representation. Otherwise, gedit compatible",
                        action='store_true', default=False)
    #  General MCTS parameters
    parser.add_argument("--itermax", help="Maximum number of tree iterations", default=1000, type=int)
    parser.add_argument("--parallel",
                        help="Using rollout parallelisation. Default is False. Should not be used at the moment.",
                        type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--expansion_width", help="Maximum number of children", default=5, type=int)
    parser.add_argument("--time_budget", help="Time budget", default=7200, type=int)
    parser.add_argument("--max_depth", help="Maximum depth of search", default=7, type=int)
    parser.add_argument("--minimal_visit_counts", default=1, type=int,
                        help="Minimal number of times a node has to be rolled out before his brothers can be expanded")
    # UCT parameters
    parser.add_argument("--UCT_policy",
                        help="UCT policy for the tree search.",
                        choices=["Classical_UCT", "Biological_UCT_1",
                                 "Classical_UCT_with_bias", "Classical_UCT_RAVE",
                                 'Biochemical_UCT_with_progressive_bias', 'Biochemical_UCT_1_with_RAVE',
                                 "Biochemical_UCT_1", "Nature_UCT", "Chemical_UCT_1",
                                 "Biological_UCT_1", "Biochemical_UCT_with_toxicity"],
                        default="Biochemical_UCT_1")
    parser.add_argument("--UCTK",
                        help="UCTK for exploration/exploitation", type=float,
                        default=20)
    parser.add_argument("--bias_k",
                        help="bias_k for exploration/exploitation", type=float,
                        default=0)
    parser.add_argument("--k_rave",
                        help="k_rave for weighting of RAVE/MCTS. Number of visits before MCTS/RAVE = 50%%", type=float,
                        default=0)
    parser.add_argument("--use_RAVE",
                        help="Use rave or not",
                        type=lambda x: (str(x).lower() == 'true'), default=False)
    # Rewarding
    parser.add_argument("--penalty",
                        help="penalty for fully unsolved state",
                        type=int, default=-1)
    parser.add_argument("--full_state_reward",
                        help="full_state_reward for fully solved state",
                        type=int, default=2)
    parser.add_argument("--pathway_scoring",
                        help="pathway scoring function",
                        choices=["RandomPathwayScorer", "constant_pathway_scoring",
                                 "null_pathway_scoring", "biological_pathway_scoring"],
                        default="constant_pathway_scoring")
    # Rollout parameters
    parser.add_argument("--Rollout_policy",
                        help="Rollout_policy for the tree search.",
                        choices=["Rollout_policy_chemical_best",
                                 "Rollout_policy_random_uniform_on_biochemical_multiplication_score",
                                 "Rollout_policy_biological_best", "Rollout_policy_biochemical_addition_best",
                                 "Rollout_policy_biochemical_multiplication_best", "Rollout_policy_random_uniform",
                                 "Rollout_policy_random_uniform_on_chem_score",
                                 "Rollout_policy_random_uniform_on_bio_score",
                                 "Rollout_policy_random_uniform_on_biochemical_addition_score", "Rollout_policy_first"],
                        default="Rollout_policy_random_uniform_on_biochemical_multiplication_score")
    parser.add_argument("--max_rollout",
                        help="Max rollout number", type=int,
                        default=3)

    # Chemical and biological scoring
    parser.add_argument("--chemical_scoring",
                        help="Chemical scoring policy.",
                        choices=["RandomChemicalScorer",
                                 "SubstrateChemicalScorer",
                                 "SubandprodChemicalScorer",
                                 "ConstantChemicalScorer"],
                        default="SubandprodChemicalScorer")
    parser.add_argument("--biological_score_cut_off", default=0.1, type=float)
    parser.add_argument("--substrate_only_score_cut_off", default=0.01, type=float)
    parser.add_argument("--chemical_score_cut_off", default=0.3, type=float)
    # Bias parameters
    parser.add_argument("--virtual_visits",
                        help="Virtual visits", type=int,
                        default=0)
    parser.add_argument("--progressive_bias_strategy",
                        help="Progressive bias strategy",
                        default="max_reward")
    parser.add_argument("--progressive_widening",
                        help="progressive_widening",
                        type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--diameter", nargs='+',
                        help="Diameters to consider", default=[2], type=int)
    parser.add_argument("--EC_filter", nargs='+',
                        help="EC numbers to consider for rules", default=None, type=str)
    parser.add_argument("--small", help="Use only a small subset", type=lambda x: (str(x).lower() == 'true'),
                        default=False)
    parser.add_argument("--seed",
                        help="Seed", type=int,
                        default=None)
    # Load from a previously run tree
    parser.add_argument("--tree_to_complete", help="Tree to restart the search from", default=None)
    parser.add_argument("--folder_tree_to_complete", help="Tree to restart the search from", default=None)

    args = parser.parse_args()
    # Config folder where to save data
    folder_to_save = define_folder_to_save(args.folder_to_save)

    if args.verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    if args.log_file is None:
        logging.basicConfig(stream=sys.stderr,
                            level=logging_level,
                            datefmt='%d/%m/%Y %H:%M:%S',
                            format='%(asctime)s -- %(levelname)s -- %(message)s')
    else:
        if not "log" in args.log_file:
            log_file = "log_" + args.log_file
        else:
            log_file = args.log_file
        log_writer = open("{}/{}".format(folder_to_save, log_file), "w")
        logging.basicConfig(stream=log_writer,
                            level=logging_level,
                            datefmt='%d/%m/%Y %H:%M:%S',
                            format='%(asctime)s -- %(levelname)s -- %(message)s')
    if not args.seed is None:
        random.seed(args.seed)
        logging.warning("Setting the seed at {}".format(args.seed))

    # Information from argparse
    representation = get_representation(args.representation)
    rules, biological_scoring = get_rules_and_score(full_rules_forward_H=full_rules_forward_H,
                                                    full_rules_retro_H=full_rules_retro_H,
                                                    full_rules_forward_no_H=full_rules_forward_no_H,
                                                    full_rules_retro_no_H=full_rules_retro_no_H,
                                                    add_Hs=add_Hs,
                                                    retro=retrosynthesis,
                                                    diameters=args.diameter,
                                                    small=args.small,
                                                    c_name=args.c_name,
                                                    filtering_EC=args.EC_filter)
    if not args.EC_filter is None:
        logging.info("Filtering rules based on EC number. Currently {} rules".format(len(rules.keys())))

    progressive_bias_strategy = get_progressive_bias(args.progressive_bias_strategy)
    # Rewarding:
    rollout_rewards = RolloutRewards(penalty=args.penalty, full_state_reward=args.full_state_reward)
    # Stop mode
    if args.stop_at_first_result:
        stop_mode_config = "Stopping at first result"
    else:
        stop_mode_config = "Full search - no stopping at first result"
    # Minimal visits
    minimal_visit_counts_config = "Setting the minimal visit count for a node at {}".format(args.minimal_visit_counts)
    # RAVE_config
    RAVE_config = "Using RAVE: {}".format(args.use_RAVE)
    # Scoring configuration
    chemical_scoring = args.chemical_scoring
    chemical_scoring_configuration = {
        "biological_score_cut_off": args.biological_score_cut_off,
        "substrate_only_score_cut_off": args.substrate_only_score_cut_off,
        "chemical_score_cut_off": args.chemical_score_cut_off}

    biological_score_config = "Using biological cut off at {}".format(args.biological_score_cut_off)
    substrate_only_score_config = "Using substrate only cut off at {}".format(args.substrate_only_score_cut_off)
    chemical_score_config = "Using chemical score cut off at {}".format(args.chemical_score_cut_off)

    # Setting chemistry info
    logging.info("Stating global parameters from configuration file")
    logging.info(tree_mode_config)
    logging.info(stop_mode_config)
    logging.info(DB_config)
    logging.info(cache_config)
    logging.info("-------------------Scoring configurations -------------------------")
    logging.info(biological_score_config)
    logging.info(substrate_only_score_config)
    logging.info(chemical_score_config)
    logging.info("-------------------MCTS configurations -------------------------")
    logging.info(minimal_visit_counts_config)
    logging.info(RAVE_config)
    logging.info("-------------------Chemical configurations -------------------------")
    logging.info(hydrogen_config)
    if use_toxicity:
        logging.info("-------------------Toxicity configurations -------------------------")
        logging.info(toxicity_scorer.log_loading)
        logging.info(toxicity_scorer.log_score)
    try:
        organism = get_organism(biosensor, organism_name = args.organism_name, complementary_sink = args.complementary_sink, add_Hs = add_Hs)
        if retrosynthesis and biosensor:
            raise RunModeError(retrosynthesis, biosensor)
        try:
            if args.tree_to_complete is None:
                root_compound = Compound(csmiles = args.c_smiles,
                                        InChI = args.c_inchi,
                                        name = args.c_name,
                                        max_moves = args.expansion_width,
                                        stereo = False,
                                        heavy_standardisation = True,
                                        fire_timeout = args.fire_timeout,
                                        chemical_scoring_configuration = chemical_scoring_configuration,
                                        standardisation_timeout = args.standardisation_timeout)
                state = ChemicalCompoundState([root_compound], representation = representation)  # state is not sanitised
                if biosensor:
                    present_in_state_detectable = organism.compound_in_state(root_compound)
                    if present_in_state_detectable:
                        logging.warning("Removed compound from the detectable set to force enzymatic detection")
                        organism.remove_cmpd_from_state(root_compound)
                else:
                    present_in_state_sink = organism.compound_in_state(root_compound)
                    if present_in_state_sink:
                        raise CompoundInSink(folder_to_save, root_compound)

                search_tree = Tree(root_state=state,
                                   itermax=args.itermax,
                                   parallel=args.parallel,
                                   available_rules=rules,
                                   rewarding=rollout_rewards,
                                   expansion_width=args.expansion_width,
                                   time_budget=args.time_budget,
                                   max_depth=args.max_depth,
                                   UCT_policy=args.UCT_policy,
                                   UCT_parameters={"UCTK": args.UCTK, "bias_k": args.bias_k, 'k_rave': args.k_rave},
                                   Rollout_policy=args.Rollout_policy,
                                   max_rollout=args.max_rollout,
                                   organism=organism,
                                   main_layer_tree=True,
                                   main_layer_chassis=True,
                                   biological_scorer=biological_scoring,
                                   chemical_scorer=chemical_scoring,
                                   folder_to_save=folder_to_save,
                                   virtual_visits=args.virtual_visits,
                                   progressive_bias_strategy=progressive_bias_strategy,
                                   progressive_widening=args.progressive_widening,
                                   heavy_saving=args.heavy_saving,
                                   minimal_visit_counts=args.minimal_visit_counts,
                                   use_RAVE=args.use_RAVE,
                                   pathway_scoring=args.pathway_scoring)
            else:
                search_tree = unpickle(file_name=args.tree_to_complete,
                                       type='tree',
                                       folder_address="{}/pickles".format(args.folder_tree_to_complete))
                # Check compound compatibility
                current_root_state = search_tree.root_state
                try:
                    root_compound = Compound(csmiles=args.c_smiles,
                                             InChI=args.c_inchi,
                                             name=args.c_name,
                                             max_moves=args.expansion_width,
                                             stereo=False,
                                             heavy_standardisation=True)
                    state = ChemicalCompoundState([root_compound],
                                                  representation=representation)  # state is not sanitised
                    if state != current_root_state:
                        raise IncorrectTreeLoading(
                            "New root {} is different from old root {} when loading tree".format(root_compound,
                                                                                                 current_root_state.compound_list[
                                                                                                     0]))
                except CompoundDefinitionException:
                    logging.warning("Use compound information from previous Tree")
                    root_compound = current_root_state
                if biosensor:
                    present_in_state_detectable = organism.compound_in_state(root_compound)
                    if present_in_state_detectable:
                        logging.warning("Removed compound from the detectable set to force enzymatic detection")
                        organism.remove_cmpd_from_state(root_compound)

                search_tree.set_heavy_saving(args.heavy_saving)
                search_tree.set_folder_to_save(folder_to_save)
                search_tree.find_full_scope(folder_to_save=folder_to_save, name="after_unpickling")
                search_tree.jsonify_full_tree(file_name="after_unpickling")

                search_tree.set_rules(rules)  # Resetting the rules to new standards and not former tree
                start_time = time.time()
                logging.info("Starting flagging for extension at {}".format(start_time))
                search_tree.flag_nodes_for_extension(extension_length=args.expansion_width,
                                             maximum_depth=args.max_depth,
                                             chemical_scoring_configuration = chemical_scoring_configuration)
                logging.info("Finished flagging for extension at {}".format(time.time() - start_time))
                search_tree.find_full_scope(folder_to_save=folder_to_save, name="after_extending_nodes")
                search_tree.jsonify_full_tree(file_name="after_extending_nodes")
            # Running the search for both trees (new or loaded)
            search_tree.run_search(args.stop_at_first_result)
        except KeyboardInterrupt as e:
            logging.warning("Keyboard interruption")

        search_tree.jsonify_full_tree()
        search_tree.find_full_scope(folder_to_save=folder_to_save)
        if args.heavy_saving:
            search_tree.find_all_scopes(folder_to_save=folder_to_save)
        try:
            search_tree.find_single_best_pathway(folder_to_save=folder_to_save)
        except IndexError:
            logging.info("No best pathway was found")
        nbr = search_tree.find_multiple_best_pathways(folder_to_save=folder_to_save, return_result=True)
        logging.info("{} pathways were found".format(nbr))
        logging.info("At the end of this tree search, the cache contains {}".format(len(home_made_cache.keys())))
        search_tree.save(folder_address=folder_to_save + '/pickles', file_name="end_search")
    except (RunModeError, IncorrectTreeLoading, CompoundInSink, InvalidSink) as e:
        logging.error(e.message)
        nbr = 0
        loading_recap = {"TIME_EXECUTION": 0, "STOP_REASON": e.reason, "NUMBER_ITERATION": 0}
        logging.warning("RECAP TIME_EXECUTION={}".format(loading_recap["TIME_EXECUTION"]))
        logging.warning("RECAP STOP_REASON={}".format(loading_recap["STOP_REASON"]))
        logging.warning("RECAP NUMBER_ITERATION={}".format(loading_recap["NUMBER_ITERATION"]))
    except ChemConversionError as e:
        logging.error(e)
        nbr = 0
        loading_recap = {"TIME_EXECUTION": 0, "STOP_REASON": "incorrect_input", "NUMBER_ITERATION": 0}
        logging.error("Verify the input SMILES or InChI is valid")
        logging.warning("RECAP TIME_EXECUTION={}".format(loading_recap["TIME_EXECUTION"]))
        logging.warning("RECAP STOP_REASON={}".format(loading_recap["STOP_REASON"]))
        logging.warning("RECAP NUMBER_ITERATION={}".format(loading_recap["NUMBER_ITERATION"]))

    # chemical_scoring_configuration = {
    #     "biological_score_cut_off": 0.1,
    #     "substrate_only_score_cut_off": 0.3,
    #     "chemical_score_cut_off": 0.3}

    # root_compound = Compound(csmiles = "CC(=CCCC(C)(C(=O)CCC1(C2CC3C1(C3C2)C)C)O)C",
    #                         InChI = "InChI=1S/C20H32O2/c1-13(2)7-6-9-19(4,22)17(21)8-10-18(3)14-11-15-16(12-14)20(15,18)5/h7,14-16,22H,6,8-12H2,1-5H3",
    #                         name = "lycosantalonol",
    #                         max_moves = 5,
    #                         stereo = False,
    #                         heavy_standardisation = True,
    #                         fire_timeout = 1,
    #                         chemical_scoring_configuration = chemical_scoring_configuration,
    #                         standardisation_timeout = 5)
    # state = ChemicalCompoundState([root_compound], representation = Test_representation)               

    # tree = Tree(root_state=state,
    #             itermax=100,
    #             expansion_width=10,
    #             time_budget=7200,
    #             max_depth=7,
    #             UCT_policy="Biochemical_UCT_1",
    #             UCT_parameters={"UCTK": 20, "bias_k": 0, 'k_rave': 0},
    #             Rollout_policy="Rollout_policy_random_uniform_on_biochemical_multiplication_score",
    #             max_rollout=3,
    #             progressive_bias_strategy=0,
    #             folder_to_save="lycosantalonol_3")
    
    # tree.run_search()