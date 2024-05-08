# Solution based on this link: https://stackoverflow.com/a/27410865
# based on Markley et al. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
import numpy
from typing import List


def average_quaternion(
    quaternions: List[List[float]]
) -> List[float]:
    quaternion_matrix = numpy.array(quaternions, dtype="float32").transpose()  # quaternions into columns
    quaternion_matrix /= len(quaternions)
    eigenvalues, eigenvectors = numpy.linalg.eig(numpy.matmul(quaternion_matrix, quaternion_matrix.transpose()))
    maximum_eigenvalue_index = numpy.argmax(eigenvalues)
    quaternion = eigenvectors[:, maximum_eigenvalue_index]
    if quaternion[3] < 0:
        quaternion *= -1
    return quaternion.tolist()
