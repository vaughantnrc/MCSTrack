import base64
import numpy
from pydantic import BaseModel, Field


class MarkerDefinition(BaseModel):
    label: str = Field()
    representation_single_base64: str = Field()  # representation from a single rotation only

    def representation_all_base64(self):
        """
        OpenCV ArUco expects to receive all possible rotations of a marker. We generate these programmatically.
        """
        representation_single_bytes: bytes = base64.b64decode(self.representation_single_base64)
        representation_single_list: list[bool] = list(representation_single_bytes)
        representation_single_matrix: numpy.ndarray = numpy.asarray(
            a=representation_single_list,
            dtype=bool)
        marker_side_length_bits: int = int(numpy.sqrt(len(representation_single_list)))
        representation_single_matrix = numpy.reshape(
            a=representation_single_matrix,
            newshape=(marker_side_length_bits, marker_side_length_bits))
        representation_all_list: list[bool] = list(representation_single_matrix.flatten())
        for i in range(3):
            representation_single_matrix = numpy.rot90(representation_single_matrix)
            representation_all_list += list(representation_single_matrix.flatten())
        representation_all_bytes: bytes = bytes(representation_all_list)
        return base64.b64encode(representation_all_bytes)
