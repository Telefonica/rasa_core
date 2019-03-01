from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import io
import logging
import typing
from collections import deque
from enum import Enum
from typing import Generator, Dict, Text, Any, Optional, Iterator
from typing import List

from mlflow.entities import Run

from rasa_core import events
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.conversation import Dialogue
from rasa_core.events import (
    UserUttered, ActionExecuted,
    Event, SlotSet, Restarted, ActionReverted, UserUtteranceReverted,
    BotUttered, Form, AllSlotsReset)
from rasa_core.slots import Slot
from rasa_core.actions import Action

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain

from rasa_core.trackers import EventVerbosity

import time
from rasa_core.utils import read_yaml_file

import traceback # Debug



PERSISTENT_SLOT_PREFIX = "P_"
EMPTY_VALUE = "__EMPTY__"

# class EventVerbosity(Enum):
#     """Filter on which events to include in tracker dumps."""
#
#     # no events will be included
#     NONE = 1
#
#     # all events, that contribute to the trackers state are included
#     # these are all you need to reconstruct the tracker state
#     APPLIED = 2
#
#     # include even more events, in this case everything that comes
#     # after the most recent restart event. this will also include
#     # utterances that got reverted and actions that got undone.
#     AFTER_RESTART = 3
#
#     # include every logged event
#     ALL = 4


class DialogueStateTracker(object):
    """Maintains the state of a conversation."""

    @classmethod
    def from_dict(cls,
                  sender_id,  # type: Text
                  events_as_dict,  # type: List[Dict[Text, Any]]
                  slots,  # type: List[Slot]
                  max_event_history=None  # type: Optional[int]
                  ):
        # type: (...) -> DialogueStateTracker
        """Create a tracker from dump.

        The dump should be an array of dumped events. When restoring
        the tracker, these events will be replayed to recreate the state."""

        evts = events.deserialise_events(events_as_dict)
        print('DIALOGUE_OK')
        return cls.from_events(sender_id, evts, slots, max_event_history)


    @classmethod
    def from_events(cls,
                    sender_id,  # type: Text
                    evts,  # type: List[Event]
                    slots,  # type: List[Slot]
                    max_event_history=None  # type: Optional[int]
                    ):
        tracker = cls(sender_id, slots, max_event_history)
        for e in evts:
#            tracker.update(e)
            tracker.update(e, online=False)  # TODO Check me!!
        return tracker

    def __init__(self, sender_id, slots,
                 max_event_history=None,
                 entity_watcher_reset_slots=None,
                 intent_watcher_context_update=None,
                 action_watcher_context_update=None,
                 persistent_slots_manager=None,
                 persistent_slots_dependencies=None):
        """Initialize the tracker.

        A set of events can be stored externally, and we will run through all
        of them to get the current state. The tracker will represent all the
        information we captured while processing messages of the dialogue."""

        # maximum number of events to store
        self._max_event_history = max_event_history
        # list of previously seen events
        self.events = self._create_events([])
        # id of the source of the messages
        self.sender_id = sender_id

        # slots that can be filled in this domain
#        self.slots = {slot.name: copy.deepcopy(slot) for slot in slots}
        # Create a Slot Dictionary, with support for persistent slots
        self.slots = SlotDictionary()
        for slot in slots:
            self.slots[slot.name] = copy.deepcopy(slot)
        self.slots.set_dialogue_state_tracker(self)

        ###
        # current state of the tracker - MUST be re-creatable by processing
        # all the events. This only defines the attributes, values are set in
        # `reset()`
        ###
        # if tracker is paused, no actions should be taken
        self._paused = None
        # A deterministically scheduled action to be executed next
        self.followup_action = ACTION_LISTEN_NAME  # type: Optional[Text]
        self.latest_action_name = None
        # Stores the most recent message sent by the user
        self.latest_message = None
        self.latest_bot_utterance = None
        self._reset()
        self.active_form = {}

        # Aura specific:
        # Initialize registries for watching entities, intents received from user, and actions.
        # This is used for automatic reset and context update from user utterances and some specific actions.
        self.entity_watcher_reset_slots = entity_watcher_reset_slots if entity_watcher_reset_slots is not None else EntityWatcher([])
        self.intent_watcher_context_update = intent_watcher_context_update if intent_watcher_context_update is not None else IntentWatcher([])
        self.action_watcher_context_update = action_watcher_context_update if action_watcher_context_update is not None else ActionWatcher([])

        # Persistent slots manager
        self.persistent_slots_manager = persistent_slots_manager
        # Add Persistent Slots Manager to Slots Dictionary
        self.slots.set_persistent_slot_manager(persistent_slots_manager)

        # This dictionary contains expiration times for persistent slots
        self.persistent_slots_expires: Dict[str, Expires] = {} # Dict[str, Expires]
        # This dictionary contains the dependecies of persistent slots on other (non persistent slots).
        # It tells when to invalidate a persistent slot value due to a change on another slot.
        self.persistent_slots_dependencies: Dict[str, List[str]] = persistent_slots_dependencies if persistent_slots_dependencies is not None else {}
        # Dictionary containing the latest slot values on which a given persistent slot depends.
        # Changing any of them will result in the invaldation of the persistent slot value and the need to read its value
        # again from external sources.
        self.persistent_slots_invalidate_dict: Dict[str, Dict[str, str]] = {}


    ###
    # Public tracker interface
    ###
    def current_state(self, event_verbosity=EventVerbosity.NONE):
        # type: (EventVerbosity) -> Dict[Text, Any]
        """Return the current tracker state as an object."""

        if event_verbosity == EventVerbosity.ALL:
            evts = [e.as_dict() for e in self.events]
        elif event_verbosity == EventVerbosity.AFTER_RESTART:
            evts = [e.as_dict() for e in self.events_after_latest_restart()]
        elif event_verbosity == EventVerbosity.APPLIED:
            evts = [e.as_dict() for e in self.applied_events()]
        else:
            evts = None

        latest_event_time = None
        if len(self.events) > 0:
            latest_event_time = self.events[-1].timestamp

        return {
            "sender_id": self.sender_id,
            "slots": self.current_slot_values(),
            "latest_message": self.latest_message.parse_data,
            "latest_event_time": latest_event_time,
            "followup_action": self.followup_action,
            "paused": self.is_paused(),
            "events": evts,
            "latest_input_channel": self.get_latest_input_channel(),
            "active_form": self.active_form,
            "latest_action_name": self.latest_action_name
        }

    def past_states(self, domain):
        # type: (Domain) -> deque
        """Generate the past states of this tracker based on the history."""

        generated_states = domain.states_for_tracker_history(self)
        return deque((frozenset(s.items()) for s in generated_states))

    def change_form_to(self, form_name):
        # type: (Text) -> None
        """Activate or deactivate a form"""
        if form_name is not None:
            self.active_form = {'name': form_name,
                                'validate': True,
                                'rejected': False}
        else:
            self.active_form = {}

    def set_form_validation(self, validate):
        # type: (bool) -> None
        """Toggle form validation"""
        self.active_form['validate'] = validate

    def reject_action(self, action_name):
        # type: (Text) -> None
        """Notify active form that it was rejected"""
        if action_name == self.active_form.get('name'):
            self.active_form['rejected'] = True

    def set_latest_action_name(self, action_name):
        # type: (Text) -> None
        """Set latest action name
            and reset form validation and rejection parameters
        """
        self.latest_action_name = action_name
        if self.active_form.get('name'):
            # reset form validation if some form is active
            self.active_form['validate'] = True
        if action_name == self.active_form.get('name'):
            # reset form rejection if it was predicted again
            self.active_form['rejected'] = False

    def current_slot_values(self):
        # type: () -> Dict[Text, Any]
        """Return the currently set values of the slots"""
        return {key: slot.value for key, slot in self.slots.items()}

    def get_slot(self, key):
        # type: (Text) -> Optional[Any]
        """Retrieves the value of a slot."""

        # if key in self.slots:
        #     _s = self.slots[key].value
        #     if key.startswith(PERSISTENT_SLOT_PREFIX):
        #             return self.persistent_slots_manager.get_slot(key, self)
        #     else:
        #         return _s
        # else:
        #     logger.info("Tried to access non existent slot '{}'".format(key))
        #     return None

        if key in self.slots:
            return self.slots[key].value
        else:
            logger.info("Tried to access non existent slot '{}'".format(key))
            return None

    def get_latest_entity_values(self, entity_type):
        # type: (Text) -> Iterator[Text]
        """Get entity values found for the passed entity name in latest msg.

        If you are only interested in the first entity of a given type use
        `next(tracker.get_latest_entity_values("my_entity_name"), None)`.
        If no entity is found `None` is the default result."""

        return (x.get("value")
                for x in self.latest_message.entities
                if x.get("entity") == entity_type)

    def get_latest_input_channel(self):
        # type: () -> Optional[Text]
        """Get the name of the input_channel of the latest UserUttered event"""

        for e in reversed(self.events):
            if isinstance(e, UserUttered):
                return e.input_channel

    def is_paused(self):
        # type: () -> bool
        """State whether the tracker is currently paused."""
        return self._paused

    def idx_after_latest_restart(self):
        # type: () -> int
        """Return the idx of the most recent restart in the list of events.

        If the conversation has not been restarted, ``0`` is returned."""

        idx = 0
        for i, event in enumerate(self.events):
            if isinstance(event, Restarted):
                idx = i + 1
        return idx

    def events_after_latest_restart(self):
        # type: () -> List[Event]
        """Return a list of events after the most recent restart."""
        return list(self.events)[self.idx_after_latest_restart():]

    def init_copy(self):
        # type: () -> DialogueStateTracker
        """Creates a new state tracker with the same initial values."""
        from rasa_core.channels import UserMessage

        return DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                    self.slots.values(),
                                    self._max_event_history)

    def generate_all_prior_trackers(self):
        # type: () -> Generator[DialogueStateTracker, None, None]
        """Returns a generator of the previous trackers of this tracker.

        The resulting array is representing
        the trackers before each action."""

        tracker = self.init_copy()

        ignored_trackers = []
        latest_message = tracker.latest_message

        for i, event in enumerate(self.applied_events()):
            if isinstance(event, UserUttered):
                if tracker.active_form.get('name') is None:
                    # store latest user message before the form
                    latest_message = event

            elif isinstance(event, Form):
                # form got either activated or deactivated, so override
                # tracker's latest message
                tracker.latest_message = latest_message

            elif isinstance(event, ActionExecuted):
                # yields the intermediate state
                if tracker.active_form.get('name') is None:
                    yield tracker

                elif tracker.active_form.get('rejected'):
                    for tr in ignored_trackers:
                        yield tr
                    ignored_trackers = []

                    if (not tracker.active_form.get('validate') or
                            event.action_name !=
                            tracker.active_form.get('name')):
                        # persist latest user message
                        # that was rejected by the form
                        latest_message = tracker.latest_message
                    else:
                        # form was called with validation, so
                        # override tracker's latest message
                        tracker.latest_message = latest_message

                    yield tracker

                elif event.action_name != tracker.active_form.get('name'):
                    # it is not known whether the form will be
                    # successfully executed, so store this tracker for later
                    tr = tracker.copy()
                    # form was called with validation, so
                    # override tracker's latest message
                    tr.latest_message = latest_message
                    ignored_trackers.append(tr)

                if event.action_name == tracker.active_form.get('name'):
                    # the form was successfully executed, so
                    # remove all stored trackers
                    ignored_trackers = []

#            tracker.update(event)
            # This is not an "online" update.
            tracker.update(event, online=False)

        # yields the final state
        if tracker.active_form.get('name') is None:
            yield tracker
        elif tracker.active_form.get('rejected'):
            for tr in ignored_trackers:
                yield tr
            yield tracker

    def applied_events(self):
        # type: () -> List[Event]
        """Returns all actions that should be applied - w/o reverted events."""

        def undo_till_previous(event_type, done_events):
            """Removes events from `done_events` until `event_type` is found."""
            # list gets modified - hence we need to copy events!
            for e in reversed(done_events[:]):
                del done_events[-1]
                if isinstance(e, event_type):
                    break

        applied_events = []
        for event in self.events:
            if isinstance(event, Restarted):
                applied_events = []
            elif isinstance(event, ActionReverted):
                undo_till_previous(ActionExecuted, applied_events)
            elif isinstance(event, UserUtteranceReverted):
                # Seeing a user uttered event automatically implies there was
                # a listen event right before it, so we'll first rewind the
                # user utterance, then get the action right before it (the
                # listen action).
                undo_till_previous(UserUttered, applied_events)
                undo_till_previous(ActionExecuted, applied_events)
            else:
                applied_events.append(event)
        return applied_events

    def replay_events(self):
        # type: () -> None
        """Update the tracker based on a list of events."""

        applied_events = self.applied_events()
        for event in applied_events:
            event.apply_to(self)

    def recreate_from_dialogue(self, dialogue):
        # type: (Dialogue) -> None
        """Use a serialised `Dialogue` to update the trackers state.

        This uses the state as is persisted in a ``TrackerStore``. If the
        tracker is blank before calling this method, the final state will be
        identical to the tracker from which the dialogue was created."""

        if not isinstance(dialogue, Dialogue):
            raise ValueError("story {0} is not of type Dialogue. "
                             "Have you deserialized it?".format(dialogue))

        self._reset()
        self.events.extend(dialogue.events)
        self.replay_events()

    def copy(self):
        """Creates a duplicate of this tracker"""
        return self.travel_back_in_time(float("inf"))

    def travel_back_in_time(self, target_time):
        # type: (float) -> DialogueStateTracker
        """Creates a new tracker with a state at a specific timestamp.

        A new tracker will be created and all events previous to the
        passed time stamp will be replayed. Events that occur exactly
        at the target time will be included."""

        tracker = self.init_copy()

        for event in self.events:
            if event.timestamp <= target_time:
#                tracker.update(event)
                tracker.update(event, online=False)
            else:
                break

        return tracker  # yields the final state

    def as_dialogue(self):
        # type: () -> Dialogue
        """Return a ``Dialogue`` object containing all of the turns.

        This can be serialised and later used to recover the state
        of this tracker exactly."""

        return Dialogue(self.sender_id, list(self.events))

    def update(self, event, online=True):
        # type: (Event) -> None
        """Modify the state of the tracker according to an ``Event``.
        :param event:
        :param online: whether this method is called with an "online" event (a just received event from user).
            This parameter is needed to know when to fire automatic slot reset and context updates.
        """

        if not isinstance(event, Event):  # pragma: no cover
            raise ValueError("event to log must be an instance "
                             "of a subclass of Event.")
        self.events.append(event)
        event.apply_to(self)


    def add_event_no_apply(self, event):
        """
        Add an event to the events list but do not execute (apply). Only register event.
        :param event:
        """
        if not isinstance(event, Event):  # pragma: no cover
            raise ValueError("event to log must be an instance "
                             "of a subclass of Event.")
        self.events.append(event)


    def export_stories(self, e2e=False):
        # type: () -> Text
        """Dump the tracker as a story in the Rasa Core story format.

        Returns the dumped tracker as a string."""
        from rasa_core.training.structures import Story

        story = Story.from_events(self.applied_events(), self.sender_id)
        return story.as_story_string(flat=True, e2e=e2e)

    def export_stories_to_file(self, export_path="debug.md"):
        # type: (Text) -> None
        """Dump the tracker as a story to a file."""

        with io.open(export_path, 'a', encoding="utf-8") as f:
            f.write(self.export_stories() + "\n")

    ###
    # Internal methods for the modification of the trackers state. Should
    # only be called by events, not directly. Rather update the tracker
    # with an event that in its ``apply_to`` method modifies the tracker.
    ###
    def _reset(self):
        # type: () -> None
        """Reset tracker to initial state - doesn't delete events though!."""

        self._reset_slots()
        self._paused = False
        self.latest_action_name = None
        self.latest_message = UserUttered.empty()
        self.latest_bot_utterance = BotUttered.empty()
        self.followup_action = ACTION_LISTEN_NAME
        self.active_form = {}

    def _reset_slots(self):
        # type: () -> None
        """Set all the slots to their initial value."""

        for slot in self.slots.values():
            slot.reset()

    def reset_slots(self):
        # type: () -> None
        """Set all the slots to their initial value."""
        self._reset_slots()

    def _set_slot(self, key, value):
        # type: (Text, Any) -> None
        """Set the value of a slot if that slot exists."""

        if key in self.slots:
            self.slots[key].value = value
        else:
            logger.error("Tried to set non existent slot '{}'. Make sure you "
                         "added all your slots to your domain file."
                         "".format(key))

    def _create_events(self, evts):
        # type: (List[Event]) -> deque

        if evts and not isinstance(evts[0], Event):  # pragma: no cover
            raise ValueError("events, if given, must be a list of events")
        return deque(evts, self._max_event_history)

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return (other.events == self.events and
                    self.sender_id == other.sender_id)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def trigger_followup_action(self, action):
        # type: (Text) -> None
        """Triggers another action following the execution of the current."""

        self.followup_action = action

    def clear_followup_action(self):
        # type: () -> None
        """Clears follow up action when it was executed"""

        self.followup_action = None

    def _merge_slots(self, entities=None):
        # type: (Optional[List[Dict[Text, Any]]]) -> List[SlotSet]
        """Take a list of entities and create tracker slot set events.

        If an entity type matches a slots name, the entities value is set
        as the slots value by creating a ``SlotSet`` event."""

        entities = entities if entities else self.latest_message.entities
        new_slots = [SlotSet(e["entity"], e["value"]) for e in entities if
                     e["entity"] in self.slots.keys()]
        return new_slots

    def update_context(self):
        """
        Updates current information context.
        :return: None
        """
        logger.debug("update_context. Implement me!!")
        # TODO Implement me or leave me abstract!

    def process_reset_slots_delta(self, event: UserUttered):
        """
        Process an event and decides whether apply a reset slots operation before processing the new event.
        The event must be of class UserUttered and the following checks are performed:
            - at least one entity ise set in last user message (UserUtter event).
            - that entity is registered in entity_watcher_reset_slots watcher.
            - the corresponding slot already has a non null value.
        :param event: UserUttered event.
        :return:
        """
        if isinstance(event, UserUttered):
            if self.entity_watcher_reset_slots.fire(event, self):
                logger.debug("Reset slots")
                # self.reset_slots()
                # # Add All Slots Reset event to the list of events, but do not execute (apply)
                # self.add_event_no_apply(AllSlotsReset())
                self.update(AllSlotsReset())

    def process_context_update(self, event):
        """
        Process an event and decides whether apply a context update operation before processing the new event.
        The event must be of class UserUttered or ActionExecuted and the following checks are performed:
            - If UserUttered:
                - an intent is identified in last user message (event UserUttered)
                - that intent is registered in intent_watcher_context_update watcher.
            - If ActionExecuted:
                - the action name is registered in action_watcher_context_update watcher.
        :param event:
        :return:
        """
        if isinstance(event, UserUttered):
            if self.intent_watcher_context_update.fire(event, self):
                logger.debug("Update context")
                self.update_context()
#        if isinstance(event, Action):
#            if self.action_watcher_context_update.fire(event, self):
#                logger.debug("Update context")
#                self.update_context()

    def process_context_update_after_action(self, action: Action):
        """
        Process an event and decides whether apply a context update operation after the execution of actions.
        :param action:
        :return:
        """
        if self.action_watcher_context_update.fire(action, self):
            logger.debug("Update context")
            self.update_context()


    def _log_slots(self):
        # Log currently set slots
        slot_values = "\n".join(["\t{}: {}".format(s.name, s.value)
                                 for s in self.slots.values()])
        logger.debug("Current slot values: \n{}".format(slot_values))



class Watcher(object):
    """
    Base class for watchers.
    """
    def __init__(self, name_register):
        self.name_register = name_register

    def fire(self, event: Event, tracker: DialogueStateTracker):
        """
        Checks whether an operation must be fired.
        :param event:
        :param tracker:
        :return: True if operation must be fired; False otherwise.
        """
        raise RuntimeError("Class Watcher is abstract. Cannot be called directly!")

    def register(self, name):
        """
        :param name:
        :return:
        """
        if not(name in self.name_register):
            self.name_register.append(name)

    def deregister(self, name):
        """
        :param name:
        :return:
        """
        if name in self.name_register:
            self.name_register.remove(name)


class EntityWatcher(Watcher):
    """
    Keeps a registry for those entities which will fire an action (such as reset slots).
    """
    def __init__(self, entities: Dict[str, List[str]]):
        super().__init__(entities)

    def fire(self, user_uttered: UserUttered, tracker: DialogueStateTracker):
        """
        Checks whether a reset slots operation must be fired.
        :param user_uttered:
        :param tracker:
        :return: True if a slots operation must be fired; False otherwise.
        """
        # Get intent
        _intent = user_uttered.intent
        # Get entities
        _ents = self._get_entities(user_uttered)
        # For every entity found
        # for _ent in _ents:
        #     if _ent[0] in self.name_register:
        #         if tracker.get_slot(_ent[0]) is not None:
        #             logger.debug("The entity {} is configured to fire reset slots. Old value is '{} 'and new value is '{}'."
        #                          " Reset slots is needed.".format(_ent[0], tracker.get_slot(_ent[0]), _ent[1]))
        #             return True
        for _ent in _ents:
            if _ent[0] in self.name_register:
                _old_slot_value = tracker.get_slot(_ent[0])
                _new_slot_value = _ent[1]
                if _intent["name"] in self.name_register[_ent[0]]:
                    if (_old_slot_value is not None) and (_old_slot_value != _new_slot_value):
                        logger.debug("The entity {} is configured to fire reset slots when intent {} is present. Old value is '{} 'and new value is '{}'."
                                     " Reset slots is needed.".format(_ent[0], _intent, tracker.get_slot(_ent[0]), _ent[1]))
                        return True
        return False

    def _get_entities(self, user_uttered: UserUttered):
        """
        Get the entities contained in a user_uttered event.
        :param user_uttered:
        :return: List[List[entity_name, entity_value]]
        """
        return [(e["entity"], e["value"]) for e in user_uttered.entities]


class IntentWatcher(Watcher):
    """
    Keeps a registry for those intents that will fire an action (such as context update).
    """
    def __init__(self, intents: List):
        super().__init__(intents)

    def fire(self, user_uttered: UserUttered, tracker: DialogueStateTracker):
        """
        Checks whether an intent operation must be fired.
        :param user_uttered:
        :param tracker:
        :return: True if a intent operation must be fired; False otherwise.
        """
        # Get intents
        _ints = self._get_intents(user_uttered)
        # For every intent found
        for _int in _ints:
            if _int in self.name_register:
                logger.debug("The intent {} is configured to fire context update."
                             " context update is needed.".format(_int))
                return True
        return False

    def _get_intents(self, user_uttered: UserUttered):
        """
        Get the entities contained in a user_uttered event.
        :param user_uttered:
        :return: List[intent_name]
        """
        if user_uttered.intent is not None:
            return [user_uttered.intent["name"]]
        return []


class ActionWatcher(Watcher):
    """
    Keeps a registry for those actions that will fire an action (such as context update).
    """
    def __init__(self, intents: List):
        super().__init__(intents)

    def fire(self, action: Action, tracker: DialogueStateTracker):
        """
        Checks whether an Action operation must be fired.
        :param action:
        :param tracker:
        :return: True if a intent operation must be fired; False otherwise.
        """
        if action.name() in self.name_register:
            logger.debug("The action {} is configured to fire context update."
                         " context update is needed.".format(action))
            return True
        return False


class PersistentSlotsManager(object):
    """
    Manages the values of the persistent slots in a conversation dialogue.
    Manages:
        - Dependencies of persistent slots on other slots
        - Expiring of values
    """

    def __init__(self, invalidate_dict_definition=None):
        """
        :param invalidate_dict: {<slot_name>: {set of slots names}}
        """
        if invalidate_dict_definition is None:
            self.invalidate_dict_definition = {}
        else:
            self.invalidate_dict_definition = invalidate_dict_definition


    @staticmethod
    def load_from_config(config_file):
        """
        Load from persistent slots configuration file.
        :param config_file:
        :return:
        """
        return PersistentSlotsManager(read_yaml_file(config_file))


    def _check_invalidate(self, slot_name, tracker: DialogueStateTracker):
        """
        Check whether a slot name current value is invalidate due to changes in its dependencies.
        :param slot_name:
        :param tracker:
        :return:
        """
        if slot_name not in tracker.persistent_slots_dependencies:
            return False
        _new_invalidate_dep = self._get_invalidate_dependencies(slot_name, tracker)
        res = _new_invalidate_dep != tracker.persistent_slots_invalidate_dict[slot_name]
        if res:
#            tracker.persistent_slots_invalidate_dict[slot_name] = _new_invalidate_dep
            logger.debug("Invalidated Persistent Slot '{}' because of changes in dependencies {} -> {}".format(slot_name,
                              tracker.persistent_slots_invalidate_dict[slot_name], _new_invalidate_dep))
        return res


    def get_slot(self, slot:Slot, tracker:DialogueStateTracker) -> Slot:
        """
        :param slot: Current local slot value
        :param tracker:
        :return: persistent_slots_invalidate_dict
        """
        _t_now = time.time()
        _invalidated = False
        if slot is None:
            _invalidated = True
        elif slot.value is None:
            _invalidated = True
        elif slot.name not in tracker.persistent_slots_expires:
            _invalidated = True
        elif tracker.persistent_slots_expires[slot.name] is None:
            _invalidated = True
        elif tracker.persistent_slots_expires[slot.name].is_expired():
            logger.debug("Expired persistent slot '{}'".format(slot.name))
            _invalidated = True
        elif self._check_invalidate(slot.name, tracker):
            _invalidated = True

#        if slot is None or (slot.name not in tracker.persistent_slots_expires) or (tracker.persistent_slots_expires[slot.name] < _t_now) or\
#                        self._check_invalidate(slot.name, tracker):
        if _invalidated:
            logger.debug("Reading persistent slot '{}' thorugh PersistentSlotsManager".format(slot.name))
            _value, _expire_seconds = self.read_slot(slot.name, tracker)
#            tracker.slots[slot.name].value = PersistentSlot(_value)
            tracker.persistent_slots_expires[slot.name] = Expires(_expire_seconds)  # Save Expires
            # Save invalidate dependencies
            tracker.persistent_slots_invalidate_dict[slot.name] = self._get_invalidate_dependencies(slot.name, tracker)
            return PersistentSlot(slot.name, _value)
        else:
            return slot


    def _get_invalidate_dependencies(self, slot_name, tracker:DialogueStateTracker):
        """
        Builds the dependencies of an external slot, as a dictionary.
        :param slot_name:
        :param tracker:
        :return: Dictionary containing the slots current persistent slot depends on and their current values.
        Any subsequent change in these values will invalidate the persistent slot value.
        """
        res = {}
        if slot_name in tracker.persistent_slots_dependencies:
            for _s_name in tracker.persistent_slots_dependencies[slot_name]:
                res[_s_name] = tracker.slots[_s_name].value
        return res


    def read_slot(self, slot_name, tracker:DialogueStateTracker):
        """
        This method gets the value of a persistent tracker, reading it from external storage / APIs.
        It must be implemented for every conversation / type of context.
        :param slot_name:
        :param tracker:
        :return: The value of the persistent slot and its expiration time: (value, expires)
        """
        raise(NotImplemented("This is an abstract class. Not yet implemented!!"))


class SlotDictionary(dict):
    """
    Special dictionary used to support persistent slots.
    """

#    __persistent_slots_manager: PersistentSlotsManager = None
#    __dialogue_state_tracker: DialogueStateTracker = None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.__persistent_slots_manager: PersistentSlotsManager = None
        self.__dialogue_state_tracker: DialogueStateTracker = None

    def set_persistent_slot_manager(self, manager: PersistentSlotsManager):
        self.__persistent_slots_manager = manager

    def set_dialogue_state_tracker(self, tracker):
        self.__dialogue_state_tracker = tracker

    # def __getitem__(self, item):
    #     _res = super().__getitem__(item)
    #     if (self.__persistent_slots_manager is not None) and item.startswith(PERSISTENT_SLOT_PREFIX) and \
    #             self.__dialogue_state_tracker is not None:
    #         _res = self.__persistent_slots_manager.get_slot(_res, self.__dialogue_state_tracker)
    #         super().__setitem__(item, _res)
    #         # Add slot set to the event list, but do not execute (apply)
    #         self.__dialogue_state_tracker.add_event_no_apply(SlotSet(item, _res.value))
    #         return _res
    #     else:
    #         return super().__getitem__(item)

    def __getitem__(self, item):
        _slot = super().__getitem__(item)
        if (self.__persistent_slots_manager is not None) and item.startswith(PERSISTENT_SLOT_PREFIX) and \
                self.__dialogue_state_tracker is not None:
            _new_slot = self.__persistent_slots_manager.get_slot(_slot, self.__dialogue_state_tracker)
            if (_slot.value != _new_slot.value):
                # The slot value has changed => update
                super().__setitem__(item, _new_slot)
                # Add slot set to the event list, but do not execute (apply)
                self.__dialogue_state_tracker.add_event_no_apply(SlotSet(item, _new_slot.value))
            return _new_slot
        else:
            return _slot

    def values(self):
        return (self.__getitem__(k) for k in self.keys())

    def items(self):
        _keys = self.keys()
        return ((k, self.__getitem__(k)) for k in self.keys())

    def get(self, k, d=None):
        if k not in self:
            return d
        return self.__getitem__(k)

    # TODO Implement setitem to allow persistence!!


class PersistentSlot(Slot):
    """
    This class represents a Persistent Slot.
    """
    type_name = "persistent"

    def as_feature(self):
#        return [1.0 if self.value is not None else 0.0]
        return [0.0 if (self.value is None or self.value == EMPTY_VALUE) else 1.0]


class Expires(object):
    """
    Expiration checking class.
    """

    def __init__(self, expire_seconds: float):
        """
        :param expire_seconds: seconds to expire from current time.
        """
        self.t_base_seconds = time.time()
        self.expire_seconds = expire_seconds

    def is_expired(self) -> bool:
        """
        :return: True if expired; False otherwise.
        """
        return time.time() > (self.t_base_seconds + self.expire_seconds)
