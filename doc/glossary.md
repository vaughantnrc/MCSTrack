# Glossary
- Annotation: A feature that is detected in an image.
- Controller: Component responsible for coordinating communication between other components.
- Detector: Component responsible for capturing images, processing them, and generating Annotations for tracking.
- Feature: Something that is identifiable either in 2D or 3D.
- Landmark: A unique feature on a Target that has its own distinct 3D coordinates.
- Pose: a position and orientation.
- Pose Solver: Component responsible for receiving Annotations and calculating Poses per Target.
- Target: A definition of something to track. Currently these consist of Landmarks.

Notes: Annotation, Feature, and Landmark are all distinct but tightly related concepts.
