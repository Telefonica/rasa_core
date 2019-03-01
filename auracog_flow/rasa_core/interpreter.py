from auracog_lib.aura_logging.logconfig import getLogger
from auracog_lib.config.auconfig import AuConfig
from auracog_lib.country_lang.country_lang import CountryLangMapper
from auracog_pipelines.interface.message import PipelineMessage
from auracog_pipelines.pipelines.final.gordon import GordonPipeline

import asyncio
import urllib.request, json
import unidecode
from rasa_core.interpreter import NaturalLanguageInterpreter
from copy import deepcopy
import logging
import time


#TEMPLATE = {"text": "", "intent": {"name": "", "confidence": 0.0}, "entities": []}
TEMPLATE = {"text": "", "intent": {"name": "", "confidence": 0.0}, "entities": [], "intent_ranking": []}


def _sanitize_entity_name(entity_name):
    """
    Sanitizes an entity name to ensure that it does not contains any dot character ('.').

    :param entity_name:
    :return:
    """
    if entity_name is None:
        return None
    return entity_name.replace(".", "_")


def _sanitize_entities(data, f=_sanitize_entity_name):
    """
    Sanitize the structure of entities returned by the server.

    :param data: data structure of the entity.
    :param f: sanitization function to be used on entity names.
    :return: the sanitized data structure.
    """
    if "entities" not in data:
        return data
    _entities = data["entities"]
    st_entities = []
    for _ent in _entities:
        _new_ent = deepcopy(_ent)
        if "entity" in _ent:
            _new_ent["entity"] = f(_ent["entity"])
        st_entities.append(_new_ent)
    result = deepcopy(data)
    result["entities"] = st_entities
    return result


class LuisHttpInterpreter(NaturalLanguageInterpreter):
    """
    Interpreter invoking to LUIS and Cognitive Services.
    Adapted from Botcorn developments.
    """

    def __init__(self, luis_api_endpoint, project="default", model="current", log=logging):
        """
        :param luis_api_endpoint:
        """
        self.luis_api_endpoint = luis_api_endpoint
        self.project = project
        self.model = model

        _request_chain = "/parse?"
        if project is not None:
            _request_chain += "project={}&".format(project)
        if model is not None:
            _request_chain += "model={}&".format(model)
        self.luis_api_parse_endpoint = luis_api_endpoint + _request_chain + "q="
        self.log = log
#        self.luis_api_parse_endpoint = luis_api_endpoint + "/parse?project={}&model={}&q=".format(project, model)


    def parse(self, text):
        """
            This is the function Rasa uses for processing messages
        """
        try:
            if len(text) == 0:
                return {}
            text = self.preprocess(text)

            result = self.query_luis_api(text)
            final = deepcopy(TEMPLATE)
            final['text'] = text
            for entity in result['entities']:
                # Set values different from entity terms
                tmp = {
                    "start": entity['startIndex'],
#                    "end": entity['endIndex'],
                    "end": entity['endIndex'] + 1,  # Needed to comply with Rasa format
                    "value": entity['entity'],
                    "entity": entity['type']
                }
                if 'score' in entity:
                    tmp['confidence'] = entity['score']
                else:
                    tmp['confidence'] = 1.0

                final['entities'].append(tmp)

            final['intent']['name'] = result['topScoringIntent']['intent']
            final['intent']['confidence'] = result['topScoringIntent']['score']

            # Add "intent_ranking"
            intent_ranking_element = []
            for intent in result["intents"]:
                intent_ranking_element.append({"name":intent["intent"],
                                               "confidence":intent["score"]})
            final["intent_ranking"] = sorted(intent_ranking_element, key=lambda element: element["confidence"], reverse=True)

            # # Optional "project" and "model" elements
            # if "project" in result:
            #     final["project"] = result["project"]
            # if "model" in result:
            #     final["model"] = result["model"]

            return _sanitize_entities(final)

        except Exception as e:
            self.log.error(str(e))
            final = deepcopy(TEMPLATE)
            final['text'] = text
            return final


    def query_luis_api(self, text):
        """
            Call luis api, gathers the intents and entities.
        """
        query = self.luis_api_parse_endpoint + str(text).replace(' ', '+')
        url = urllib.request.urlopen(query)
        data = json.loads(url.read().decode())
        return data


    def get_norm(self, entities):
        """
            Search the maximum score among entities
        """
        t = 0
        for value in entities:
            if value['score'] > t:
                t = value['score']
        return t


    def sort_by_score(self, cognitive):
        result = sorted(cognitive, key=lambda element: element['score'], reverse=True)
        return result


    def preprocess(self, text):
        """
            Preprocess sentences in order to erase all problematic characters

            TODO Check me!
        """
        _text = unidecode.unidecode(text).strip()
        if _text[0] in ['/', '#', '?', '¿', '@']:
            _text = _text[1:]
        if '/' in _text:
            _text = _text.replace('/', '')
        return _text


class AuraPipelineInterpreter(NaturalLanguageInterpreter):
    """
    Interpreter invoking Aura pipeline.
    """

    def __init__(self, au_cfg, log, channel = 'mh', lang_country = 'es-es'):
        """
        :param au_config: Auconfig
        :param log:
        :param channel:
        :param lang_country:
        """
        # Create pipeline
        self.pipeline = GordonPipeline(cfg=au_cfg, log=log)
        self.country_lang = CountryLangMapper.parse_country_lang(cfg=au_cfg, log=log, lang_country=lang_country)
        self.log = log
        self.channel = channel


    def parse(self, text, channel=None):
        """
            This is the function Rasa uses for processing messages
        """
        try:
            pipeline_message = PipelineMessage(phrase=text)

            loop = asyncio.get_event_loop()

            _channel = self.channel if channel is None else channel
            start_time = time.time()
            status = loop.run_until_complete(
                self.pipeline.run(pipeline_msg=pipeline_message, country_lang=self.country_lang, channel=_channel))

            end_time = time.time()
            self.log.debug("Parsing time: {}".format(end_time - start_time))

            if not status:
                self.log.error("Error parsing {}".format(text))

            # Build result
            result = deepcopy(TEMPLATE)
            result["text"] = text
            # Add entities
            for entity in pipeline_message.get_entities():
                # Set values different from entity terms
                tmp = {
                    "start": entity.get_start_index(),
                    "end": entity.get_end_index(),
                    "value": entity.get_name(),
                    "entity": entity.get_entity_type()
                }
                tmp['confidence'] = entity.get_score()
                result['entities'].append(tmp)

            #
            result["intent"]["name"] = pipeline_message.get_intent()
            result["intent"]["confidence"] = pipeline_message.get_intent_score()

            # Add "intent_ranking"
            # TODO check me!
            intent_ranking_element = []
#            for intent in result["intents"]:
#                intent_ranking_element.append({"name": intent["intent"],
#                                               "confidence": intent["score"]})
            intent_ranking_element.append({"name": pipeline_message.get_intent(),
                                               "confidence": pipeline_message.get_intent_score()})
            result["intent_ranking"] = sorted(intent_ranking_element, key=lambda element: element["confidence"],
                                             reverse=True)

            return _sanitize_entities(result)

        except Exception as e:
            self.log.error(str(e))
            final = deepcopy(TEMPLATE)
            final['text'] = text
            return final


if __name__ == "__main__":
    # Testing code

#    REQUEST_TEXT = "quiero ver el segundo episodio de la temporada 3 de star trek"
    REQUEST_TEXT = "Quiero ver algún episodio de Star Trek"
#    REQUEST_TEXT = "Quiero ver Star Wars"

    LUIS_API_ENDPOINT = "http://localhost:5000"

    INTERPRETER_TYPE = "AuraPipeline"
#    INTERPRETER_TYPE = "LUIS"

    channel = 'mh'
    lang_country = 'es-es'
    au_config_path = "."
    cfg = AuConfig('config_auracog_flow', path=au_config_path, interpolation=None)
    log = getLogger('stub-pipelines', config=cfg, docker=False)

    if INTERPRETER_TYPE == "LUIS":
        interpreter = LuisHttpInterpreter(LUIS_API_ENDPOINT, log=log)
    else:
        interpreter = AuraPipelineInterpreter(cfg, log, channel = channel, lang_country = lang_country)

    res = interpreter.parse(REQUEST_TEXT)
    print("Request text: {}\nResult:\n{}".format(REQUEST_TEXT, json.dumps(res, indent=4)))

