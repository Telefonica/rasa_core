# Code to support rules engines integration into DialogueStateTracker
#

from rasa_core.utils import class_from_module_path
from auracog_flow.rasa_core.trackers import DialogueStateTracker
from pyknow import KnowledgeEngine
import logging
import sys
from tqdm import tqdm
import threading

#from auracog_flow.series_dialogue.trackers import ContextReasoner

logger = logging.getLogger(__name__)

class TrackerKnowledgeEngine(KnowledgeEngine):

    def __init__(self):
        super().__init__()
        self.tracker = None

    def run_with_tracker(self, tracker: DialogueStateTracker, steps: float):
        """
        Run the rules engine using a DialogueStateTracker.
        :param tracker:
        :param steps: Maximum number of reasoning steps to be performed.
        :return:
        """
        self.tracker = tracker
        super().run(steps=steps)


class PyKnowRulesEnginePool(object):
    """
    Rules engine pool. Singleton.
    """

    class __PyKnowRulesEnginePool(object):

        def __init__(self, reasoner_class: str, initial_pool_size: int=10):
            """
            :param reasoner_class:
            :param initial_pool_size:
            """
            self.pool_size = 10
            self.reasoner_class = reasoner_class
            # Reasoner class resolved
            self.reasoner_class = reasoner_class
            logger.info("Creating rules engine pool with size={}".format(initial_pool_size))
            self.engines = [class_from_module_path(self.reasoner_class)() for _ in tqdm(range(initial_pool_size))]
            logger.info("Rules engine pool created [{}]".format(sys.getsizeof(self)))
            # Lock to support multithreading
            self.lock = threading.RLock()
            #
            self.busy = []

        def acquire_rules_engine(self) -> TrackerKnowledgeEngine:
            """
            Acquires a rules engine object. Thread safe.
            :return:
            """
            engine = None
            with self.lock:
                self.lock.acquire()
                if len(self.engines) > 0:
                    engine = self.engines.pop(-1)
                self.lock.release()
            if engine is None:
                engine = class_from_module_path(self.reasoner_class)()
            with self.lock:
                self.lock.acquire()
                self.busy.append(engine)
                self.lock.release()
            logger.debug("Acquired rules engine: {}".format(engine))
            return engine

        def release_rules_engine(self, engine: KnowledgeEngine):
            """
            Releases a rules engine object. Thread safe.
            :param engine:
            :return:
            """
            with self.lock:
                self.lock.acquire()
                self.busy.pop(self.busy.index(engine))
                self.engines.append(engine)
                self.lock.release()
            logger.debug("Released rules engine: {}".format(engine))

    instance: __PyKnowRulesEnginePool = None

    def __init__(self, reasoner_class: str, initial_pool_size: int=10):
        """
        :param reasoner_class:
        :param initial_pool_size:
        """
        if PyKnowRulesEnginePool.instance is None:
            PyKnowRulesEnginePool.instance = PyKnowRulesEnginePool.__PyKnowRulesEnginePool(reasoner_class, initial_pool_size)

    def acquire_rules_engine(self) -> TrackerKnowledgeEngine:
        """
        Acquires a rules engine object. Thread safe.
        :return:
        """
        return PyKnowRulesEnginePool.instance.acquire_rules_engine()

    def release_rules_engine(self, engine: TrackerKnowledgeEngine) -> None:
        """
        Releases a rules engine object. Thread safe.
        :param engine:
        :return:
        """
        PyKnowRulesEnginePool.instance.release_rules_engine(engine)
