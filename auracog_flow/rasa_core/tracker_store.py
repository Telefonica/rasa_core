from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import typing

import json
import logging

# noinspection PyPep8Naming
import six.moves.cPickle as pickler
from typing import Text, Optional, List

from rasa_core.utils import class_from_module_path
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.broker import EventChannel
from auracog_flow.rasa_core.trackers import(
    DialogueStateTracker, ActionExecuted,
    EventVerbosity, IntentWatcher, EntityWatcher, ActionWatcher)

import yaml

#from auracog_flow.series_dialogue.persistent_slots import SeriesPersistentSlotsManager  # Debug


logger = logging.getLogger()

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain


DEFAULT_DIALOGUE_STATE_TRACKER_CLASS = "auracog_flow.rasa_core.trackers.DialogueStateTracker"

class TrackerStore(object):
    def __init__(self, domain, event_broker=None,
                 entities_to_reset_slots=[], intents_to_update_context=[],
                 actions_to_update_context=[], dialogue_state_tracker_class=DEFAULT_DIALOGUE_STATE_TRACKER_CLASS,
                 persistent_slots_manager=None, persistent_slots_dependencies={}):
        """
        :param domain:
        :param event_broker:
        :param entities_to_reset_slots: List of entities firing reset slots
        :param intents_to_update_context: List of intents firing update context
        :param actions_to_update_context: List of actions firing update context
        :param dialogue_state_tracker_class: Module name and class of the DialogueStateTracker. Default is
            'auracog_flow.rasa_core.trackers.DialogueStateTracker'
        :param persistent_slots_manager: Persistent slots read/write manager.
        """
        # type: (Optional[Domain], Optional[EventChannel]) -> None
        self.domain = domain
        self.event_broker = event_broker
        # Aura specific:
        # Dictionary of entities firing reset slots: Dict[str, List[str]] (Dict[entity_name, List[intent_name])
        self.entities_to_reset_slots = entities_to_reset_slots
        # List of intents firing update context: List[str] (List[intent_name])
        self.intents_to_update_context = intents_to_update_context
        # List of actions firing update context
        self.actions_to_update_context = actions_to_update_context
        #
        self.dialogue_state_tracker_class = dialogue_state_tracker_class
        #
        self.persistent_slots_manager = persistent_slots_manager
        # dictionary of persistent slots dependencies
        self.persistent_slots_dependencies = persistent_slots_dependencies

    @staticmethod
    def find_tracker_store(domain, store=None, event_broker=None, context_config_file=None, persistent_slots_config_file=None):
        """
        :param domain:
        :param store:
        :param event_broker:
        :param context_config_file: YAML file containing entities and intents to be watched.
        :param persistent_slots_config_file: YAML file containing persistent slots dependencies configuration data.
        :return:
        """
        # Aura specific ####
        _entities_to_reset_slots = []
        _intents_to_update_context = []
        _actions_to_update_context = []
        _persistent_store_manager = None
        _dialogue_state_tracker_class = DEFAULT_DIALOGUE_STATE_TRACKER_CLASS
        _persistent_slots_manager = None
        _persistent_slots_dependencies = {}
        if context_config_file is not None:
            with open(context_config_file, "r") as f:
                config_data = yaml.load(f)
            _dialogue_state_tracker_class = config_data.get("dialogue_state_tracker_class", [DEFAULT_DIALOGUE_STATE_TRACKER_CLASS])[0]
            _entities_to_reset_slots = config_data.get("entities_to_reset_slots", {})
            _intents_to_update_context = config_data.get("intents_to_update_context", [])
            _actions_to_update_context = config_data.get("actions_to_update_context", [])
            _persistent_slots_manager_class = config_data.get("persistent_slots_manager_class", [None])[0]
            if _persistent_slots_manager_class is not None:
                _persistent_slots_manager = class_from_module_path(_persistent_slots_manager_class)()

        if persistent_slots_config_file is not None:
            with open(persistent_slots_config_file) as f:
                _persistent_slots_dependencies = yaml.load(f)
        ####

        if store is None or store.store_type is None:
            res = InMemoryTrackerStore(domain, event_broker=event_broker)
        elif store.store_type == 'redis':

            res = RedisTrackerStore(domain=domain,
                                     host=store.url,
                                     event_broker=event_broker,
                                     **store.kwargs)
        elif store.store_type == 'mongod':
            res = MongoTrackerStore(domain=domain,
                                     host=store.url,
                                     event_broker=event_broker,
                                     **store.kwargs)
        else:
            res = TrackerStore.load_tracker_from_module_string(domain, store)
        # Aura specific:
        # Set intents and entities firing special behaviour
        res.dialogue_state_tracker_class = _dialogue_state_tracker_class
        res.entities_to_reset_slots = _entities_to_reset_slots
        res.intents_to_update_context = _intents_to_update_context
        res.actions_to_update_context = _actions_to_update_context
        res.persistent_slots_manager = _persistent_slots_manager
        res.persistent_slots_dependencies = _persistent_slots_dependencies
        logger.debug("dialogue_state_tracker_class: {}".format(_dialogue_state_tracker_class))
        logger.debug("entities_to_reset_slots: {}; intents_to_update_context: {}; actions_to_update_context: {}; persistent_slots_manager: {}".format(_entities_to_reset_slots,
                       _intents_to_update_context, _actions_to_update_context, _persistent_slots_manager))
        ####
        return res

    @staticmethod
    def create_tracker_from_module_string(module_class_name, sender_id, slots, entity_watcher_reset_slots=None,
                                          intent_watcher_context_update=None, action_watcher_context_update=None,
                                          persistent_slots_manager=None, persistent_slots_dependencies=None):
        """
        Creates a tracker from its module and class name.
        :param module_class_name:
        :param sender_id:
        :param slots:
        :param entity_watcher_reset_slots:
        :param intent_watcher_context_update:
        :param action_watcher_context_update:
        :param persistent_slots_manager:
        :param persistent_slots_dependencies
        :return:
        """
        tracker_class = class_from_module_path(module_class_name)
        return tracker_class(sender_id, slots, entity_watcher_reset_slots=entity_watcher_reset_slots,
                             intent_watcher_context_update=intent_watcher_context_update,
                             action_watcher_context_update=action_watcher_context_update,
                             persistent_slots_manager=persistent_slots_manager,
                             persistent_slots_dependencies=persistent_slots_dependencies)

    @staticmethod
    def load_tracker_from_module_string(domain, store):
        custom_tracker = None
        try:
            custom_tracker = class_from_module_path(store.store_type)
        except (AttributeError, ImportError):
            logger.warning("Store type {} not found. "
                           "Using InMemoryTrackerStore instead"
                           .format(store.store_type))

        if custom_tracker:
            return custom_tracker(domain=domain,
                                  url=store.url, **store.kwargs)
        else:
            return InMemoryTrackerStore(domain)

    def get_or_create_tracker(self, sender_id):
        tracker = self.retrieve(sender_id)
        if tracker is None:
            tracker = self.create_tracker(sender_id)
        return tracker

    # def init_tracker(self, sender_id):
    #     if self.domain:
    #         return DialogueStateTracker(sender_id,
    #                                     self.domain.slots,
    #                                     entity_watcher_reset_slots=EntityWatcher(self.entities_to_reset_slots),
    #                                     intent_watcher_context_update=IntentWatcher(self.intents_to_update_context),
    #                                     action_watcher_context_update=ActionWatcher(self.actions_to_update_context))
    #     else:
    #         return None

    def init_tracker(self, sender_id):
        """
        :param module_class_name:
        :param sender_id:
        :return:
        """
        if self.domain:
            return TrackerStore.create_tracker_from_module_string(self.dialogue_state_tracker_class, sender_id,
                                                                  self.domain.slots,
                                                                  entity_watcher_reset_slots=EntityWatcher(self.entities_to_reset_slots),
                                                                  intent_watcher_context_update=IntentWatcher(self.intents_to_update_context),
                                                                  action_watcher_context_update=ActionWatcher(self.actions_to_update_context),
                                                                  persistent_slots_manager=self.persistent_slots_manager,
                                                                  persistent_slots_dependencies=self.persistent_slots_dependencies)
        else:
            return None


    def create_tracker(self, sender_id, append_action_listen=True):
        """Creates a new tracker for the sender_id.

        The tracker is initially listening."""

        tracker = self.init_tracker(sender_id)
        if tracker:
            if append_action_listen:
                tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
            self.save(tracker)
        return tracker

    def save(self, tracker):
        raise NotImplementedError()

    def retrieve(self, sender_id):
        # type: (Text) -> Optional[DialogueStateTracker]
        raise NotImplementedError()

    def stream_events(self, tracker):
        # type: (DialogueStateTracker) -> None
        old_tracker = self.retrieve(tracker.sender_id)
        offset = len(old_tracker.events) if old_tracker else 0
        evts = tracker.events
        for evt in list(itertools.islice(evts, offset, len(evts))):
            body = {
                "sender_id": tracker.sender_id,
            }
            body.update(evt.as_dict())
            self.event_broker.publish(json.dumps(body))

    def keys(self):
        # type: () -> Optional[List[Text]]
        raise NotImplementedError()

    @staticmethod
    def serialise_tracker(tracker):
#        dialogue = tracker.as_dialogue()
#        return pickler.dumps(dialogue)
#        Tracker in Aura becomes much more complex, so serialization must preserve all data.
        logger.debug("Serialising tracker for uiserid {}".format(tracker.sender_id))
        return pickler.dumps(tracker)

    def deserialise_tracker(self, sender_id, _json):
#        dialogue = pickler.loads(_json)
#        tracker = self.init_tracker(sender_id)
#        tracker.recreate_from_dialogue(dialogue)
#        return tracker
        # Tracker in Aura becomes much more complex, so serialization must preserve all data.
        tracker = pickler.loads(_json)
        logger.debug("Deserialised tracker for uiserid {}".format(tracker.sender_id))
        return tracker


class InMemoryTrackerStore(TrackerStore):
    def __init__(self, domain, event_broker=None):
        self.store = {}
        super(InMemoryTrackerStore, self).__init__(domain, event_broker)

    def save(self, tracker):
        if self.event_broker:
            self.stream_events(tracker)
        serialised = InMemoryTrackerStore.serialise_tracker(tracker)
        self.store[tracker.sender_id] = serialised

    def retrieve(self, sender_id):
        if sender_id in self.store:
            logger.debug('Recreating tracker for '
                         'id \'{}\''.format(sender_id))
            return self.deserialise_tracker(sender_id, self.store[sender_id])
        else:
            logger.debug('Creating a new tracker for '
                         'id \'{}\'.'.format(sender_id))
            return None

    def keys(self):
        return self.store.keys()


class RedisTrackerStore(TrackerStore):
    def keys(self):
        pass

    def __init__(self, domain, host='localhost',
                 port=6379, db=0, password=None, event_broker=None,
                 record_exp=None):

        import redis
        self.red = redis.StrictRedis(host=host, port=port, db=db,
                                     password=password)
        self.record_exp = record_exp
        super(RedisTrackerStore, self).__init__(domain, event_broker)

    def save(self, tracker, timeout=None):
        if self.event_broker:
            self.stream_events(tracker)


        if not timeout and self.record_exp:
            timeout = self.record_exp

        serialised_tracker = self.serialise_tracker(tracker)
        self.red.set(tracker.sender_id, serialised_tracker, ex=timeout)


    def retrieve(self, sender_id):
        stored = self.red.get(sender_id)
        if stored is not None:
            return self.deserialise_tracker(sender_id, stored)
        else:
            return None


class MongoTrackerStore(TrackerStore):
    def __init__(self,
                 domain,
                 host="mongodb://localhost:27017",
                 db="rasa",
                 username=None,
                 password=None,
                 auth_source="admin",
                 collection="conversations",
                 event_broker=None):
        from pymongo.database import Database
        from pymongo import MongoClient

        self.client = MongoClient(host,
                                  username=username,
                                  password=password,
                                  authSource=auth_source,
                                  # delay connect until process forking is done
                                  connect=False)

        self.db = Database(self.client, db)
        self.collection = collection
        super(MongoTrackerStore, self).__init__(domain, event_broker)

        self._ensure_indices()

    @property
    def conversations(self):
        return self.db[self.collection]

    def _ensure_indices(self):
        self.conversations.create_index("sender_id")

    def save(self, tracker, timeout=None):
        if self.event_broker:
            self.stream_events(tracker)

        state = tracker.current_state(EventVerbosity.ALL)

        self.conversations.update_one(
                {"sender_id": tracker.sender_id},
                {"$set": state},
                upsert=True)

    def retrieve(self, sender_id):
        stored = self.conversations.find_one({"sender_id": sender_id})

        # look for conversations which have used an `int` sender_id in the past
        # and update them.
        if stored is None and sender_id.isdigit():
            from pymongo import ReturnDocument
            stored = self.conversations.find_one_and_update(
                {"sender_id": int(sender_id)},
                {"$set": {"sender_id": str(sender_id)}},
                return_document=ReturnDocument.AFTER)

        if stored is not None:
            if self.domain:
                return DialogueStateTracker.from_dict(sender_id,
                                                      stored.get("events"),
                                                      self.domain.slots)
            else:
                logger.warning("Can't recreate tracker from mongo storage "
                               "because no domain is set. Returning `None` "
                               "instead.")
                return None
        else:
            return None

    def keys(self):
        return [c["sender_id"] for c in self.conversations.find()]
