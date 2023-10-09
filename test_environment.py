import os
import numpy as np
import pickle
import copy

import environment.env_config as env_config


def generate_test_sets(num, config: env_config.EnvConfig):
    # generate and store test sets for arena, eval etc
    # we do this always for the same seed
    np.random.seed(42)
    arena_instances = []
    eval_instances = []
    test_instances = []

    # for some situations, we want to test the agent on feeds provided by literature (of
    # course only if these are considered in this training process):
    # Acetone Chloroform: equimolar, wang2018
    # Water Ethanol: equimolar, kunnakorn2013
    # Butanol Water: 0.4 But, 0.6 W, luyben2008
    # Water Pyridine: 0.1 P, 0.9 W, chen2015
    steps = 50
    if config.systems_allowed["acetone_chloroform"]:
        temp_arena, temp_eval, temp_test = helper_test_set_generation(
            names=["acetone", "chloroform"], config=config, steps=steps,
            set_feeds=[[np.array([0.5, 0.5, 0])]])

        arena_instances = arena_instances + temp_arena
        eval_instances = eval_instances + temp_eval
        test_instances = test_instances + temp_test

    if config.systems_allowed["ethanol_water"]:
        temp_arena, temp_eval, temp_test = helper_test_set_generation(
            names=["ethanol", "water"], config=config, steps=steps,
            set_feeds=[[np.array([0.5, 0.5, 0])]])

        arena_instances = arena_instances + temp_arena
        eval_instances = eval_instances + temp_eval
        test_instances = test_instances + temp_test

    if config.systems_allowed["n-butanol_water"]:
        temp_arena, temp_eval, temp_test = helper_test_set_generation(
            names=["n-butanol", "water"], config=config, steps=steps,
            set_feeds=[[np.array([0.4, 0.6, 0])]])

        arena_instances = arena_instances + temp_arena
        eval_instances = eval_instances + temp_eval
        test_instances = test_instances + temp_test

    if config.systems_allowed["water_pyridine"]:
        temp_arena, temp_eval, temp_test = helper_test_set_generation(
            names=["water", "pyridine"], config=config, steps=steps,
            set_feeds=[[np.array([0.9, 0.1, 0])]])

        arena_instances = arena_instances + temp_arena
        eval_instances = eval_instances + temp_eval
        test_instances = test_instances + temp_test

    # create random problem instances to store
    for i in range(num - len(eval_instances)):
        arena_instances.append(config.create_random_problem_instance())
        eval_instances.append(config.create_random_problem_instance())

    pickle.dump(arena_instances, open(os.path.join(os.getcwd(), "test", "syngaz_arena.pickle"), "wb"))
    pickle.dump(eval_instances, open(os.path.join(os.getcwd(), "test", "syngaz_eval.pickle"), "wb"))
    pickle.dump(test_instances, open(os.path.join(os.getcwd(), "test", "syngaz_test.pickle"), "wb"))


def helper_test_set_generation(names, config, steps, set_feeds=None):
    arena_instances = []
    eval_instances = []
    test_instances = []

    # set_feeds is a list of lists, containing preset feed stream lists
    if set_feeds is not None:
        for feeds in set_feeds:
            instance = find_sit_create_instance(spec_feeds=feeds,
                                                names_comps=names,
                                                config=config)

            arena_instances.append(instance)
            eval_instances.append(instance)
            test_instances.append(instance)

    for i in range(steps - 1):
        instance = find_sit_create_instance(
            spec_feeds=[np.array([(i + 1) * (1 / steps), 1 - ((i + 1) * (1 / steps)), 0])],
            names_comps=names,
            config=config)

        test_instances.append(instance)

    return arena_instances, eval_instances, test_instances


def find_sit_create_instance(spec_feeds, names_comps, config: env_config.EnvConfig):
    for sit_ind in range(len(config.phase_eq_generator.feed_situations)):
        if len(config.phase_eq_generator.feed_situations[sit_ind][0]) == len(names_comps):
            all_in = True
            for name in names_comps:
                if not config.phase_eq_generator.names_components.index(name) in \
                       config.phase_eq_generator.feed_situations[sit_ind][0]:
                    all_in = False
                    break

            if all_in:
                index = sit_ind
                break

    situation = copy.deepcopy(config.phase_eq_generator.feed_situations[index])

    # we do not shuffle in this case
    # get names in feed streams
    names_in_streams = []
    for i in situation[0]:
        names_in_streams.append(config.phase_eq_generator.names_components[i])

    instance = {"feed_situation_index": index,
                "indices_components_in_feeds": situation[0],
                "list_feed_streams": spec_feeds,
                "possible_ind_add_comp": situation[1],
                "comp_order_feeds": names_in_streams,
                "lle_for_start": None,
                "vle_for_start": None}

    return instance
