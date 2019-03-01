from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect

from flask import Blueprint, jsonify, request, Flask, Response

from rasa_core import utils
from rasa_core.constants import DOCS_BASE_URL

try:
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin

from rasa_core.channels import InputChannel
from rasa_core.channels.channel import QueueOutputChannel, UserMessage, CollectingOutputChannel
from threading import Thread
import json

from typing import Text

class AuraRestInput(InputChannel):
    """
    A custom http input channel used for integration into Aura.
    """

    @classmethod
    def name(cls):
        return "aura_rest"

    @staticmethod
    def on_message_wrapper(on_new_message, text, queue, sender_id):
        collector = QueueOutputChannel(queue)

        message = UserMessage(text, collector, sender_id,
                              input_channel=AuraRestInput.name())
        on_new_message(message)

        queue.put("DONE")

    def _extract_sender(self, req):
        return req.json.get("sender", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req):
        """
        Expected format of message:
            {'sender': 'default',
             'message': 'pon la temporada 3 de star trek',
             'parse_data': {
                'intent': {
                'name': 'episode_season_request',
                'confidence': 0.6703092455863953
                 },
                'entities': [{'start': 17, 'end': 18, 'value': '3', 'entity': 'num_season', 'confidence': 0.8917057052629874},
                            {'start': 22, 'end': 31, 'value': 'star trek', 'entity': 'tef_audiovisual_tvseries_title', 'confidence': 0.9707644335686035}
                ],
                'intent_ranking': [{'name': 'episode_season_request', 'confidence': 0.6703092455863953},
                                {'name': 'tef.int.es_ES.mh.tv.launch', 'confidence': 0.48097896575927734},
                                {'name': 'provide_content_name', 'confidence': 0.28564441204071045},
                                {'name': 'confirm', 'confidence': 0.0},
                                {'name': 'tef.int.es_ES.mh.tv.display', 'confidence': 0.0},
                                {'name': 'negation', 'confidence': 0.0},
                                {'name': 'thanks', 'confidence': 0.0},
                                {'name': 'greet', 'confidence': 0.0},
                                {'name': 'exit', 'confidence': 0.0},
                                {'name': 'number', 'confidence': 0.0}
                ]
             }
            }

        :returns (<message>, <parse_data>)
        """
        return (req.json.get("message", None), req.json.get("parse_data", None))

    def stream_response(self, on_new_message, text, sender_id):
        from multiprocessing import Queue

        q = Queue()

        t = Thread(target=self.on_message_wrapper,
                   args=(on_new_message, text, q, sender_id))
        t.start()
        while True:
            response = q.get()
            if response == "DONE":
                break
            else:
                yield json.dumps(response) + "\n"

    def blueprint(self, on_new_message):
        custom_webhook = Blueprint(
                'custom_webhook_{}'.format(type(self).__name__),
                inspect.getmodule(self).__name__)

        @custom_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @custom_webhook.route("/webhook", methods=['POST'])
        def receive():
            sender_id = self._extract_sender(request)
            text, parse_data = self._extract_message(request)
            should_use_stream = utils.bool_arg("stream", default=False)

            if should_use_stream:
                return Response(
                        self.stream_response(on_new_message, text, sender_id),
                        content_type='text/event-stream')
            else:
                # collector = CollectingOutputChannel()
                collector = CollectingCommandOutputChannel()
                on_new_message(UserMessage(text,
                                           output_channel=collector,
                                           sender_id=sender_id,
                                           parse_data=parse_data,
                                           input_channel=self.name()))
                return jsonify(collector.messages)

        return custom_webhook


class CollectingCommandOutputChannel(CollectingOutputChannel):
    """ Output channel that collects send messages in a list
    (doesn't send them anywhere, just collects them). It collects the following messages and actions:
    - Messages to user
    - End of dialogue action (action_end_of_dialogue)
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def name(cls):
        return "collector_aura"

    # @staticmethod
    # def _message(recipient_id,
    #              text=None,
    #              image=None,
    #              buttons=None,
    #              attachment=None):
    #     """Create a message object that will be stored."""
    #
    #     obj = {
    #         "recipient_id": recipient_id,
    #         "text": text,
    #         "image": image,
    #         "buttons": buttons,
    #         "attachment": attachment
    #     }
    #
    #     # filter out any values that are `None`
    #     return utils.remove_none_values(obj)

    @staticmethod
    def _message_command(recipient_id,
                         command_text):
        obj = {
            "recipient_id": recipient_id,
            "command": command_text
        }
        return obj

    def send_command(self, recipient_id, command):
        """
        Send a command over the channel, with the following format:
        {
            target_id: <user_id>,
            command: <command text>
        }
        :param recipient_id:
        :param command:
        """
        self._persist_message(self._message_command(recipient_id, command))

# def latest_output(self):
    #     if self.messages:
    #         return self.messages[-1]
    #     else:
    #         return None
    #
    # def _persist_message(self, message):
    #     self.messages.append(message)
    #
    # def send_text_message(self, recipient_id, message):
    #     for message_part in message.split("\n\n"):
    #         self._persist_message(self._message(recipient_id,
    #                                             text=message_part))
    #
    # def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
    #     self._persist_message(self._message(recipient_id,
    #                                         text=message,
    #                                         buttons=buttons))
    #
    # def send_image_url(self, recipient_id: Text, image_url: Text) -> None:
    #     """Sends an image. Default will just post the url as a string."""
    #
    #     self._persist_message(self._message(recipient_id,
    #                                         image=image_url))
    #
    # def send_attachment(self, recipient_id: Text, attachment: Text) -> None:
    #     """Sends an attachment. Default will just post as a string."""
    #
    #     self._persist_message(self._message(recipient_id,
    #                                         attachment=attachment))



