import json

import uvicorn
from transformers import AutoModelForCausalLM, StoppingCriteriaList, MaxLengthCriteria
from fastapi import FastAPI, Request, APIRouter
import sys
import logging
import experimental_samplers
from experimental_samplers import *

import back_end_utils


class Backend_Server:
    def __init__(self):
        """
        :name __init__ - intializes the Backend_Server's data members and attaches API routers to their async functions.
        """
        self.tokenizer = None
        self.model = None
        self.device = None
        self.experimental_warpers = []
        self.experimental_processors = []
        self.router = APIRouter()
        self.router.add_api_route("/load", self.load, methods=["POST"])
        self.router.add_api_route("/generate", self.generate, methods=["POST"])

    async def load(self, a_request: Request):
        """
        :name load - loads the model as per requested from the front end.
        :param request: the request from the front end which holds the data needed for loading.
        """
        loading_settings = await a_request.json()
        self.device = loading_settings['device']
        try:
            model_settings = self.replace_symbolic_data_type(loading_settings['model_settings'])
            self.load_model_settings(model_settings)
        except Exception as loading_exception:
            logging.error("There was an error loading the model. Please check the following log:")
            logging.exception(loading_exception)

    def replace_symbolic_data_type(self, a_model_settings: dict):
        """
        :name replace_symbolic_data_type - replaces a string version of the data type of the model with its corresponding
        torch data type.
        :param model_settings: The settings related to the model.
        :type dict
        :return: dict, representing the updated model settings
        """
        # Exists because there are some values we need use values known to the interpreter rather than in string format.
        # an example is the fact that while the user might specify 'float16', we need to cast this value to actually
        # be the value of torch.float16 to work.
        if "model_data_type" in a_model_settings:
            model_data_type = a_model_settings.pop("model_data_type")

            if model_data_type == "float32":
                a_model_settings['torch_dtype'] = torch.float32
            elif model_data_type == "float16":
                a_model_settings['torch_dtype'] = torch.float16
            elif model_data_type == "int8":
                # Loading in 8 bit is rather unique in the sense that you can't pass a torch data type. Instead,
                # you must pass an additional argument, as it's (supposedly) not standard yet, and uses 8bit adam.
                a_model_settings['load_in_8bit'] = True
            elif model_data_type == "int4":
                # int4 is similar to int8, ableit with even worse support. I haven't been able to find anything in
                # regards to it getting transformers library support quite yet.
                logging.error(
                    "int4 is currently not supported by the transformers library! Please use only the following data "
                    "types: float32, float16, or int8.")
            else:
                logging.error("Unknown data type! Please use only the following data types: float32, float16, or int8.")
        return a_model_settings

    def load_model_settings(self, a_model_settings: dict):
        """
        :name load_model_settings - extracts the model settings from the parameter model_settings and puts it into the
        backend_server class.
        :param model_settings: the model settings from which to extract from.
        :type dict
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(**(a_model_settings))
            print("Model loaded.")
        else:
            logging.warning(f"You already have a model loaded!")

    def initialize_experimental_warpers(self, a_experimental_warper_settings):
        """
        :name initialize_experimental_warpers - populate the experimental_warpers list data member with various
        experimental logit warper classes.
        :description This function primarily populates the experimental_warpers list data members with custom predefined
        warpers which inherit from the transformer's library LogitWarper. In this case, it is the warpers, top_a,
        and tfs.
        :param experimental_warper_settings: The settings relating to experimental warper settings, which determine
        what experimental warpers will be added to the experimental_warpers list data member.
        """
        for warper, passed_val in a_experimental_warper_settings.items():
            if warper == "top_a":
                self.experimental_warpers.append(experimental_samplers.TopALogitsWarper(float(passed_val)))
            elif warper == "tfs":
                self.experimental_warpers.append(experimental_samplers.TailFreeSamplingLogitsWarper(float(passed_val)))
            else:
                raise ValueError(f"There is no experimental warper called {warper} implemented!")

    def initialize_experimental_processors(self, a_experimental_processor_settings):
        """
        :name initialize_experimental_processors - populate the experimental_processor list data member with various
        experiment logit processor classes.
        :description This function primarily populates the experimental_processor
        list data members with custom predefined processor which inherit from the transformer's library
        LogitProcessor. In this case, only logit_bias is implemented, but more may soon come. Due to the complications
        surrounding serializing tuples, most of the oncoming processor data will be read as a 2d list. Hence, the
        algorithm will be as such:
        1. Loop through each of the individual processors
        2. In the case of logit_bias, loop through each of the 2d list and instead make it as a dict as tuples
        can be seen as 'pairs', or keys and values.
        3. Gradually append the 'key' and 'values' to a list, which is then added to the experimental_processors
           list data member.
        :param experimental_processor_settings: The settings relating to experimental
        processor settings, which determine what experimental processor will be added to the experimental_processor
        list data member.
        """
        tuple_list = []
        for processor in a_experimental_processor_settings:
            if processor == "logit_bias":
                for word, bias in dict(a_experimental_processor_settings[processor]).items():
                    tuple_list.append((word, float(bias)))
        if tuple_list is not None:
            self.experimental_processors.append(LogitBiasProcessor(tuple_list))

    def dispatch_experimental_warpers(self, a_input_ids, a_generation_settings):
        """
        :name dispatch_experimental_warpers - goes through each item of the experimental_warpers list data member
        and samples the chat history in input_ids utilizing said experimental warpers.
        :description: This function primarily goes through the experimental_warpers list data member, and for each,
        utilizing the chat history represented in input_ids as its main parameter, it applies some transformations
        utilizing said experimental warpers. For instance, top_a would take only up to a certain percentage of tokens
        and gets rid of the rest depending on their probability squared and the passed threshold. The sampling itself
        is achieved through model.sample rather than model.generate, as model.generate does not have a logits_warper
        parameter whereas model.sample does. However, model.sample has its own quirks like being needed to pass a
        stopping_criteria. See comments below.
        :param a_input_ids: A tensor representing the tokenized chat history before or after sampling. This is so the
        model can directly apply the experimental logit warpers.
        :type torch.Tensor, of varying type depending on data type.
        :param a_generation_settings: The settings from which to extract certain settings to aid in sampling with
            experimental logit warpers, such as max_length.
        :return: torch.Tensor, representing the tokenized list of the chat_history.
        """
        max_length = a_generation_settings['max_length'] if 'max_length' in a_generation_settings.keys() else None
        eos_token_id = a_generation_settings['eos_token_id'] if 'eos_token_id' in a_generation_settings.keys() else None
        pad_token_id = a_generation_settings['pad_token_id'] if 'pad_token_id' in a_generation_settings.keys() else None
        # Unfortunately, if you need to use max_length with model.sample, you need to set it up as a stopping_criteria
        # instead, as passing max_length directly is deprecated.
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=max_length if max_length is not None else 300)])

        for experimental_warper in self.experimental_warpers:
            # We use model.sample as model.generate doesn't have a logits_warper argument, just a logits_processor argument.
            # I attempted to pass logit warpers to logits_processor, and it had 0 effect.
            # I will need to run further experiments to see if this is still the case.
            a_input_ids = self.model.sample(stopping_criteria=stopping_criteria, eos_token_id=eos_token_id,
                                            pad_token_id=pad_token_id, input_ids=a_input_ids,
                                            logits_warper=experimental_warper)
        return a_input_ids

    async def generate(self, a_request: Request):
        """
        :name generate- generates text given text input which is extracted from the user's server request.
        :param request: The user's request, which has several pieces of information critical to the model's text
        generation.
        :description: This function is the bread and butter of this program, the backbone, etc.. This function responds
        to the user requests to endpoint /generate, which is used the most often. Here is the basic steps of how it
        works:
        1. The request is retrieved, and the settings are each assigned to different variables.
        2. Experimental warpers and processors are extracted from settings.
        3. The chat history is decoded and translated to the appropriate device on the backend.
        4. Text is generated through model.generate, which takes in the generation settings, and experimental
           processors are dispatched as well. After the text is generated, a slicing operation is done just to get
           purely just the generated text back.
        5. Experimental warpers are intialized and dispatched and further sample the text.
        6. All experimental warpers and processors are cleared from dispatch.
        7. The message is encoded into base 64 and returned via a server response.
        :type: startlette.Request
        :return: a base 64 encoded string representing the generated text after tokenization.
        """
        # Retrieval of the request and intialization of settings.
        decoded_json = json.loads(await a_request.json())
        gen_settings = decoded_json['generation_settings']
        device = decoded_json['device']
        experimental_settings = decoded_json[
            'experimental_settings'] if "experimental_settings" in decoded_json.keys() else None

        # Extraction of logits warper and processor settings from experimental settings
        experimental_warper_settings = None
        experimental_processor_settings = None
        if experimental_settings is not None:
            experimental_warper_settings = experimental_settings[
                'experimental_warpers'] if 'experimental_warpers' in experimental_settings.keys() else None
            experimental_processor_settings = experimental_settings[
                'experimental_processors'] if 'experimental_processors' in experimental_settings.keys() else None

        # decoding of the chat history
        tensor_chat_history = back_end_utils.decode_encoded_tokenized_tensor(decoded_json['chat_history'])
        input_ids = tensor_chat_history.to(device)

        # Intialization and dispatch of experimental logits warpers and processors
        if experimental_processor_settings is not None and len(experimental_processor_settings) > 0:
            self.initialize_experimental_processors(experimental_processor_settings)
        if experimental_warper_settings is not None and len(experimental_warper_settings) > 0:
            self.initialize_experimental_warpers(experimental_warper_settings)

        bot_message = ""
        try:
            # Generation of text and experimental processors. While blocking and constrained by the GPU,
            # the main problem is not necessarily blocking, as the max_time parameters solves that issue,
            # but the fact of the message being potentially empty.
            bot_message = self.model.generate(input_ids, **gen_settings, logits_processor=self.experimental_processors)
        except Exception as generation_exception:
            logging.error("There was an exception raised during generation:")
            logging.exception(generation_exception)

        if bot_message is not None:
            bot_message = bot_message[:, input_ids.shape[1]:]
            try:
                bot_message = self.dispatch_experimental_warpers(bot_message, gen_settings)
            except Exception as experimental_warper_exception:
                logging.error("There was an exception raised during the dispatching of experimental warpers:")
                logging.exception(experimental_warper_exception)
            return back_end_utils.get_encoded_str_from_token_list(bot_message)
        else:
            logging.warning("The generated message was empty!")
            return None

        self.experimental_processors.clear()
        self.experimental_warpers.clear()


def main():
    """
    :name - main. The main driver of the back-end. Handles starting up the server on a certain port.
    """

    try:
        port = 8000 if len(sys.argv) < 2 else int(sys.argv[1])
    except ValueError:
        logging.error("Port cannot be non-numeric!")
    except Exception as port_error:
        logging.error("There was a problem with setting the port. Please check the following log:")
        logging.exception(port_error)

    app = FastAPI()
    server = Backend_Server()
    app.include_router(server.router)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
