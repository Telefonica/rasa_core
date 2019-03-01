#
# Normalizing taks of user messages
#

from copy import deepcopy

from rasa_core.channels import UserMessage
from rasa_core.domain import Domain
from .constants import RESERVED_INTENTS, ENTITY_UNKNOWN, INTENT_UNKNOWN

from typing import Dict
import logging

logger = logging.getLogger(__name__)

class UserMessageNormalizer(object):
    """
    This class normalizes a user message, looking for unknown intents and entities according to a given domain.
    If not present in this domain, entities and intents are labeled as UNKNOWN.
    """

    @classmethod
    def normalize(cls, user_message: UserMessage, domain: Domain) -> UserMessage:
        """
        :param user_message:
        :param domain:
        :return: Normalized user message.
        """
        res = deepcopy(user_message)
        res.parse_data = UserMessageNormalizer.normalize_parsedata(res.parse_data, domain)
        return res


    @classmethod
    def normalize_parse_data(cls, parse_data: Dict, domain: Domain) -> UserMessage:
        """
        :param parse_data:
        :param domain:
        :return: Normalized parse data.
        """
        res = deepcopy(parse_data)
        # Normalize intent
        if "intent" in res:
            _intent = res["intent"]
            if _intent["name"] not in RESERVED_INTENTS:  # Do not modify RESERVED INTENTS (such as /restart or /stop)
                if _intent["name"] not in domain.intents:
                    _intent["original_name"] = _intent["name"]
                    _intent["name"] = INTENT_UNKNOWN
        # Normalize all intents in intent_ranking
        if "intent_ranking" in res:
            _intent_ranking = res["intent_ranking"]
            for _i in _intent_ranking:
                if _i["name"] not in RESERVED_INTENTS:  # Do not modify RESERVED INTENTS (such as /restart or /stop)
                    if _i["name"] not in domain.intents:
                        _i["original_name"] = _i["name"]
                        _i["name"] = INTENT_UNKNOWN
        # Normalize entities
        if "entities" in res:
            _entities = res["entities"]
            for _e in _entities:
                if _e["entity"] not in domain.entities:
                    _e["original_entity"] = _e["entity"]
                    _e["entity"] = ENTITY_UNKNOWN
        logger.debug("Normalized user message: {}".format(res))
        return res

