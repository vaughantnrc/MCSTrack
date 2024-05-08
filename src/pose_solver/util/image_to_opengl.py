import numpy


def transformation_image_to_opengl(
    transformation_matrix_image: numpy.ndarray
) -> numpy.ndarray:
    transformation_matrix_detector = numpy.array(transformation_matrix_image)
    transformation_matrix_detector[1:3, :] = -transformation_matrix_detector[1:3, :]
    return transformation_matrix_detector
    # Equivalent to:
    # transformation_matrix_180 = numpy.identity(4, dtype="float")
    # transformation_matrix_180[1, 1] *= -1
    # transformation_matrix_180[2, 2] *= -1
    # return numpy.matmul(transformation_matrix_180, transformation_matrix_image)


def vector_image_to_opengl(
    vector_image: numpy.ndarray | list[float]
) -> numpy.ndarray:
    vector_detector = numpy.array(vector_image)
    vector_detector[1:3] = -vector_detector[1:3]
    return vector_detector
    # Equivalent to:
    # transformation_matrix_180 = numpy.identity(4, dtype="float")
    # transformation_matrix_180[1, 1] *= -1
    # transformation_matrix_180[2, 2] *= -1
    # return numpy.matmul(transformation_matrix_180, vector_image)
