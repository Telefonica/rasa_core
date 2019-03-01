import json
from numpy.random import randint
import tqdm
import logging

class LUISTrainingFilesProcessor(object):
    """
    Process a LUIS training data file in tef format and produces a transformed file.
    """
    def __init__(self):
        pass

    def generate_training_data(self, dict_items_path, training_set_path,
                               luis_schema_version, version_id,
                               culture, date, format, name, schema, version,
                               random_seed=13, num_repetitions=1):
        """

        :param dict_items_path: dictionary items path.
        :param training_set_path: training set path in tef format.
        :param luis_schema_version:
        :param version_id:
        :param culture:
        :param date:
        :param format:
        :param name:
        :param schema:
        :param version:
        :param num_repetitions: number of samples to genearate for every example utterance.
        :return:
        """
        # Load dict items
        with open(dict_items_path) as f:
            dict_items = json.load(f)

        # Load training set
        with open(training_set_path) as f:
            training_set = json.load(f)

        res = {
            "luis_schema_version": luis_schema_version,
            "versionId": version_id,
            "name": name,
            "desc": '{' + '"culture": "{}", "date": "{}", "format": "{}", "name": "{}", "schema": "{}", "version": "{}"'.format(
                culture, date, format, name, schema, version)  + '}',
            "culture": culture,
            "intents":[],
            "composites": [],
            "closedLists": [],
            "bing_entities": [],
            "actions": [],
            "model_features": [],
            "regex_features": [],
            "regex_entities": [],
            "patterns": [],
            "utterances": []
        }

        # Write intent list
        res["intents"] = [{"name":intent_name} for intent_name in training_set["intents"]]

        # Write entities
        res["entities"] = self._get_unique_entities(training_set)

        # Write utterances
        for intent, intent_ex_list in tqdm.tqdm(training_set["intents"].items()):
            for intent_ex in intent_ex_list:
                for i in range(num_repetitions):
                    text, entities = self._replace_entities(intent_ex, dict_items)
                    utter = {"text": text,
                             "entities": entities,
                             "intent": intent}
                    res["utterances"].append(utter)
        return res


    def _replace_entities(self, intent_ex, dict_items, num_repetitions=1):
        """
        Replace entities with samples randomly taken from the dict_items dictionary.
        :param intent_ex: text of the intent (intent example).
        :param dict_items:
        :param num_repetitions:
        :return: (<text>, <list_dicts_entities>)
        """
        # Get the entities
        list_dicts_entities = []
        text = intent_ex
        entities, _, indexes = self._get_entities_from_str(intent_ex)
        for i in range(num_repetitions):
            offset = 0
            for i, ent in enumerate(entities):
                ent_sample = self._sample_entity(dict_items, ent)[0]
                start_index, end_index = indexes[i]
                text = text[:start_index + offset] + ent_sample + text[end_index + 1 + offset:]
                list_dicts_entities.append({
                    "entity": ent[1:-1],
                    "startPos": start_index + offset,
                    "endPos": end_index + offset + len(ent_sample) - len(ent)
                })
                offset += len(ent_sample) - len(ent)
        return (text, list_dicts_entities)


    def _get_unique_entities(self, training_set):
        """
        Get the unique entities in a training set dictionary.
        :param training_set:
        :return: List of found entities
        """
        res = {}
        for intent, intent_ex_list in training_set["intents"].items():
            for intent_ex in intent_ex_list:
                entities_b, entities, positions = self._get_entities_from_str(intent_ex)
                for ent in entities:
                    if ent not in res:
                        res[ent] = {
                            "name": ent,
                            "roles": []
                        }
        return [v for v in res.keys()]


    def _get_entities_from_str(self, s):
        """
        Get the entities contained in a string (delimited by "[" and "]").
        :param s:
        :return: ([entity_with_brackets*], [entity*], [(start_position, end_position)*])
        """
        _s = s
        res = ([], [], [])
        offset = 0
        while "[" in _s:
            start_index = _s.index("[")
            _s = _s[start_index:]
            if "]" in _s:
                end_index = _s.index("]")
                entity = _s[:end_index+1]
                res[0].append(entity)
                res[1].append(entity[1:-1])
                res[2].append((offset + start_index, offset + start_index + end_index))
                _s = _s[end_index+1:]
                offset += start_index + end_index + 1
            else:
                break
        return res


    def _sample_entity(self, dict_items, entity_name, size=1):
        """
        Get a random entity from an items dictionary.
        :param dict_items:
        :param entity_name:
        :param size:
        :return: [sample_entity+]
        """
        if entity_name[0] == "[" and entity_name[-1] == "]":
            _entity_name = entity_name[1:-1]
        else:
            _entity_name = entity_name
        if _entity_name in dict_items:
            _entities = dict_items[_entity_name]
        else:
            logging.warning("{} not in items dictionary. Leaving unchanged.".format(_entity_name))
            _entities = [entity_name]*size
        return [_entities[i] for i in randint(0, len(_entities), size=size)]


# TODO Create command line interface fpr ease of use.
if __name__ == "__main__":
    TRAIN = False

    DICT_ITEMS_PATH = "NLU/datasets/sdict_items.json"
    TRAINING_SET_PATH = "NLU/datasets/luis_trainingset.tv_content.tef.json"

    DESTINATION_PATH_TRAIN = "NLU/datasets/gen/luis_trainingset.tv_dialogues.full.json"
    DESTINATION_PATH_TEST = "NLU/datasets/gen/luis_testset.tv_dialogues.full.json"
    DESTINATION_PATH = DESTINATION_PATH_TRAIN if TRAIN else DESTINATION_PATH_TEST

    NUM_REPETITONS = 4

    LUIS_SCHEMA_VERSION = "2.1.0"
    VERSION_ID = "dev"
    CULTURE = "es-es"
    DATE = "2019-01-21"
    FORMAT = "tef:training:intent"
    NAME = "AURA Movistar Home tv R5"
    SCHEMA = "2.1"
    VERSION = "dev"

    RANDOM_SEED_TRAIN = 13
    RANDOM_SEED_TEST = 14

    RANDOM_SEED = RANDOM_SEED_TRAIN if TRAIN else RANDOM_SEED_TEST


    processor = LUISTrainingFilesProcessor()

    res = processor.generate_training_data(DICT_ITEMS_PATH,
                                           TRAINING_SET_PATH,
                                           LUIS_SCHEMA_VERSION,
                                           VERSION_ID,
                                           CULTURE,
                                           DATE,
                                           FORMAT,
                                           NAME,
                                           SCHEMA,
                                           VERSION,
                                           num_repetitions=NUM_REPETITONS,
                                           random_seed=RANDOM_SEED)

    with open(DESTINATION_PATH, "w") as f:
        f.write(json.dumps(res, indent=4, ensure_ascii=False))



