"""
Loads full pickles saved before the modules were imported through their packages. Those name
their classes by bare module ('prioritized_memory.Experience' in a replay buffer of a full-state
archive); this maps them to the package paths. Pass the module as torch.load's pickle_module.
"""
import pickle

LEGACY_MODULES = {
    **{m: f'environment.{m}' for m in ('action_encoding', 'battle', 'board', 'curriculum', 'game_state',
                                       'heuristic_filter', 'heuristic_setup', 'league', 'observation',
                                       'parallel_environment', 'piece', 'scenario_drills')},
    **{m: f'network.{m}' for m in ('distributional_reward', 'drqn_agent', 'exploiter_agents',
                                   'history_aggregator', 'opponents', 'prioritized_memory', 'networks',
                                   'aaren', 'training')},
    'training_config': 'settings.training_config',
    'policy_search': 'utils.policy_search',
    'training_utils': 'utils.training_utils',
}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        top, dot, rest = module.partition('.')
        if top in LEGACY_MODULES:
            module = LEGACY_MODULES[top] + dot + rest
        return super().find_class(module, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def loads(data, **kwargs):
    import io
    return Unpickler(io.BytesIO(data), **kwargs).load()
