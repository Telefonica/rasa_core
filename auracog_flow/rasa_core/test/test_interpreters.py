#
# Test Intepreters, including integration with the NLP pipeline.
#
from auracog_lib.aura_logging.logconfig import getLogger
from auracog_lib.config.auconfig import AuConfig
from auracog_lib.country_lang.country_lang import CountryLangMapper
from auracog_pipelines.interface.message import PipelineMessage
from auracog_pipelines.pipelines.final.gordon import GordonPipeline

from typing import Dict, List, Text
from auracog_flow.rasa_core.interpreter import AuraPipelineInterpreter
import pprint


TESTING_TEXTS = [
    "quiero que pongas vikingos",
    "ponme la segunda temporada de star trek"
]
LUIS_API_ENDPOINT = "http://localhost:5001"
INTERPRETER_TYPE = "AuraPipeline"

CHANNEL = "mh"
LANG_COUNTRY = "es-es"

AU_CONFIG_PATH = "."
CFG = AuConfig('config_auracog_flow', path=AU_CONFIG_PATH, interpolation=None)
LOG = getLogger('stub-pipelines', config=CFG, docker=False)


def test_aura_pipeline_interepreter(texts: List[Text], channel="mh", lang_country="es-es"):
    """
    :param texts:
    :return:
    """
    interpreter = AuraPipelineInterpreter(CFG, LOG, channel=channel, lang_country=lang_country)
    _res = [interpreter.parse(t) for t in texts]
    return _res


if __name__ == "__main__":
    # test_aura_pipeline_interepreter
    res = test_aura_pipeline_interepreter(TESTING_TEXTS)
    pprint.pprint(res)




