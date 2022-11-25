from compreface.common.typed_dict import ExpandedOptionsDict
from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection

DOMAIN: str = 'http://localhost'
PORT: str = '8000'
RECOGNITION_API_KEY: str = '36aee6be-da90-412f-95ac-9fc706775567'

compre_face: CompreFace = CompreFace(DOMAIN, PORT, {
    "det_prob_threshold": 0.8
})

recognition: RecognitionService = compre_face.init_face_recognition(
    RECOGNITION_API_KEY)

face_collection: FaceCollection = recognition.get_face_collection()

# Image from local path.
image: str = 'C:/Users/Abuelgasim/Downloads/images/john-wick.jpg'
subject: str = 'Test Subject'

print(face_collection.add(image, subject))
