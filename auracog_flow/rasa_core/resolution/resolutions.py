import requests
import logging
from rasa_core import utils
import pprint

logger = logging.getLogger(__name__)

#
# TODO Make behaviour asynchronous to allow multiple requests to different endpoints in parallel.
#

# HTTP API messages
OPERATION_EXECUTE = "execute"
JSON_ACTION_RESTART = {
	"action": "action_restart"
}
JSON_ACTION_LISTEN = {
	"action": "action_listen"
}
OPERATION_RETRIEVE_TRACKER = "tracker"

# RestApi messages
SEND_MESSAGE = {
  "sender": None,
  "message": None
}

class Resolver(object):

    def __init__(self, endpoints):
        """
        :param endpoints: Dict{String(conversation domain): List(Rasa RestAPI endpoint, Rasa HTTP API server)}
        """
        self.endpoints = endpoints

    def resolve_user_message(self, text):
        """
        Resolves which conversation is the most suitable for a given text.
        :param text: Text to analyze.
        :return: ResolutionResult
        """
        # Get test user
        test_user = self.get_test_user()
        trackers_info = {}
        for domain_name,endpoints in self.endpoints.items():
            # The operations in this loop should be executed synchronously (one after another, in the same order as
            # they are stated). However, the execution of the different steps of the loop can be parallelized and
            # can be executed asynchronously --> TODO
            _restapi_endpoint, _http_api_endpoint = endpoints
            self.restart_tracker(_http_api_endpoint, test_user)
            self.force_listen(_http_api_endpoint, test_user)
            self.send_message_to_restapi(text, _restapi_endpoint, test_user)
            trackers_info[domain_name] = self.retrieve_tracker(_http_api_endpoint, test_user)
        return self._resolve_user_message_from_trackers_info(trackers_info)

    def get_test_user(selfself):
        """
        Gets a safe user_id for conversation resolution. This user id must not be used for any other purpose, cannot be
        concurrently used by two parallel execution flows and shall not correspond to a real user id.
        :return:
        """
        # TODO Implement me!
        return "test_user"


    def _resolve_user_message_from_trackers_info(self, trackers_info):
        """
        Resolve from the information contained in a dictionary of trackers.
        :param trackers_info: Dict{String(conversation domain), Dict(tracker information)}
        :return: ResolutionResult
        """
        conversation_domains_scores_nlu = {}
        conversation_domains_scores_core = {}
        nlu_analysis = {}
        predicted_actions = {}
        for domain_name,tracker_info in trackers_info.items():
            conversation_domains_scores_nlu[domain_name] = tracker_info["events"][1]["parse_data"]["intent"]["confidence"]  # 0 position is occupied by action_listen
            conversation_domains_scores_core[domain_name] = self._get_min_prob_actions(tracker_info)
            nlu_analysis[domain_name] = tracker_info["events"][1]["parse_data"]  # 0 position is occupied by action_listen
            predicted_actions[domain_name] = self._get_actions(tracker_info)
        return ResolutionResult(conversation_domains_scores_nlu, conversation_domains_scores_core, nlu_analysis, predicted_actions)

    def _get_actions(self, tracker_info):
        """
        Gets an ordered list with the events of type action contained in a tracker information.
        :param tracker_info:
        :return: List(Dict(event info))
        """
        return [ev for ev in tracker_info["events"] if ev["event"] == "action"]

    def _get_min_prob_actions(self, tracker_info):
        """
        Gets the minimum probability of actions within tracker information.
        If an action corresponds to fallback, its probability is considered as 0.
        :param tracker_info:
        :return: Float
        """
        res = 1
        for a in self._get_actions(tracker_info):
            if self._is_fallback(a):
                return 0
            _p = a.get("confidence", 1)
            if _p is not None:
                res = min(res, _p)
        return res

    def _is_fallback(self, action_event):
        """
        Gets whether an action can be considered as a fallback action.
        :param action_event: Dict(action event)
        :return: Boolean
        """
        _name = action_event["name"].lower()
        return "fallback" in _name

    def restart_tracker(self, http_api_endpoint, user_id):
        """
        Restart the tracker corresponding to a user at a given endpoint.
        :param http_api_endpoint:
        :param user_id:
        :return: None
        """
        res = requests.post(self._get_url(http_api_endpoint, user_id, OPERATION_EXECUTE), json=JSON_ACTION_RESTART)
        logger.debug('restart_tracker response("{}", "{}"): {}'.format(http_api_endpoint, user_id, res))
        logger.debug('restart_tracker response("{}", "{}"): {}'.format(http_api_endpoint, user_id, res.text))

    def force_listen(self, http_api_endpoint, user_id):
        """
        Restart the tracker corresponding to a user at a given endpoint.
        :param http_api_endpoint:
        :param user_id:
        :return: None
        """
        res = requests.post(self._get_url(http_api_endpoint, user_id, OPERATION_EXECUTE), json=JSON_ACTION_LISTEN)
        logger.debug('force_listen response("{}", "{}"): {}'.format(http_api_endpoint, user_id, res))
        logger.debug('force_listen response("{}", "{}"): {}'.format(http_api_endpoint, user_id, res.text))

    def send_message_to_restapi(self, message_text, restapi_endpoint, user_id):
        """
        Sends a message to a the restapi for a given user_id.
        :param message_text:
        :param restapi_endpoint:
        :param user_id:
        :return: None
        """
        _msg = SEND_MESSAGE.copy()
        _msg["sender"] = user_id
        _msg["message"] = message_text
        res = requests.post(restapi_endpoint, json=_msg)
        logger.debug('send_message_to_rest_api response("{}", "{}", "{}"): {}'.format(message_text, restapi_endpoint, user_id, res))
        logger.debug('send_message_to_rest_api response("{}", "{}", "{}"): {}'.format(message_text, restapi_endpoint, user_id, res.text))

    def retrieve_tracker(self, http_api_endpoint, user_id):
        """
        Gets the tracker information for a user from a given HTTP API endpiont.
        :param _http_api_endpoint:
        :param test_user:
        :return: Dictionary containing the tracker information.
        """
        res = requests.get(self._get_url(http_api_endpoint, user_id, OPERATION_RETRIEVE_TRACKER))
        logger.debug('retrieve_tracker("{}", "{}") response: {}'.format(http_api_endpoint, user_id, res))
        logger.debug('retrieve_tracker("{}", "{}") response: {}'.format(http_api_endpoint, user_id, res.text))
        return res.json()

    def _get_url(self, endpoint, user_id, operation):
        """
        Build the URL corresponding to an endpoint, user_id and operation.
        :param endpoint:
        :param user_id:
        :param operation:
        :return: String(url)
        """
        _endpoint = endpoint
        while _endpoint.endswith("/"):
            _endpoint = _endpoint[:-2]
        return _endpoint + "/" + user_id + "/" + operation


class ResolutionResult(object):
    """
    Resolution result.
    """
    def __init__(self, conversation_domains_scores_nlu,
                 conversation_domains_scores_core,
                 nlu_analysis, predicted_actions):
        """
        :param conversation_domains_scores_nlu: Dict{String(conversation domain): Float(best intent nlu score)}
        :param conversation_domains_scores_core: Dict{String(conversation domain): Float(highest next action probability)}
        :param nlu_analysis: Dict{String(conversation domain): Dict{parsed data}}
        :param predicted_actions: Dict{String(conversation domain): List(Dict(action event info))}
        """
        self.conversation_domains_scores_nlu = conversation_domains_scores_nlu
        self.conversation_domains_scores_core = conversation_domains_scores_core
        self.nlu_analysis = nlu_analysis
        self.predicted_actions = predicted_actions

    def __str__(self):
        return "ResolutionResult[conversation_domains_scores_nlu: {}, conversation_domains_scores_core: {}, " \
               "nlu_analysis: {}, predicted_actions: {}]".format(self.conversation_domains_scores_nlu,
                                                                 self.conversation_domains_scores_core,
                                                                 self.nlu_analysis,
                                                                 self.predicted_actions)

    def pprint(self):
        print("ResolutionResult")
        print("- conversation_domains_scores_nlu:")
        pprint.pprint(self.conversation_domains_scores_nlu)
        print("- conversation_domains_scores_core:")
        pprint.pprint(self.conversation_domains_scores_core)
        print("- nlu_analysis:")
        pprint.pprint(self.nlu_analysis)
        print("- predicted_actions:")
        pprint.pprint(self.predicted_actions)

def __test():
    """
    TEST
    """

    logger.setLevel(logging.DEBUG)
    utils.configure_colored_logging(logging.DEBUG)

    # endpoints: Dict{String(conversation domain): List(Rasa RestAPI endpoint, Rasa HTTP API server)}
    endpoints = {
        "series": ["http://localhost:5002/webhooks/rest/webhook",
                   "http://localhost:5002/conversations"]
    }
    #msg = "pon el episodio 3 de la temporada 5 de star trek"
    #msg = "quiero ver star trek"
    #msg = "esto es una prueba negativa"
    #msg = "pon la temporada tres de star trek"
    msg = "pon la temporada 3 de doctor en alaska"
    resolver = Resolver(endpoints)

    res = resolver.resolve_user_message(msg)

    pprint.pprint(res)
    print(res)
    res.pprint()


if __name__ == "__main__":
    __test()
