import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http
import numpy as np

class wrapped_triton:
  def __init__(self, model_url: str, ) -> None:
    fullprotocol, location = model_url.split("://")
    _, protocol = fullprotocol.split("+")
    address, model, version = location.split("/")

    self._protocol = protocol
    self._address = address
    self._model = model
    self._version = version

    # check connection to server, throw error if connection doesn't work
    if self._protocol == "grpc":
      self._client = triton_grpc.InferenceServerClient(url=self._address,
                                                       verbose=False,
                                                       ssl=True)
      self._triton_protocol = triton_grpc
    elif self._protocol == "http":
      self._client = triton_http.InferenceServerClient(url=self._address,
                                                       verbose=False,
                                                       concurrency=12,
                                                       )
      self._triton_protocol = triton_http
    else:
      raise ValueError(
          f"{self._protocol} does not encode a valid protocol (grpc or http)")

  def __call__(self, input_dict, output_name) -> np.ndarray:
    '''
    Run inference of model on triton server
    '''

    # put inputs in proper format
    inputs = []
    for key in input_dict:
      input = self._triton_protocol.InferInput(key, input_dict[key].shape,
                                               "FP32")
      input.set_data_from_numpy(input_dict[key])
      inputs.append(input)

    output = self._triton_protocol.InferRequestedOutput(output_name)

    # make request to server for inference
    request = self._client.infer(self._model,
                                 model_version=self._version,
                                 inputs=inputs,
                                 outputs=[output],
                                 )
    out = request.as_numpy(output_name)

    return out