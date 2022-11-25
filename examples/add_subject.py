from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import Subjects

DOMAIN: str = 'http://localhost'
PORT: str = '8000'
RECOGNITION_API_KEY: str = '36aee6be-da90-412f-95ac-9fc706775567'

compre_face: CompreFace = CompreFace(DOMAIN, PORT)

recognition: RecognitionService = compre_face.init_face_recognition(RECOGNITION_API_KEY)

subjects: Subjects = recognition.get_subjects()

subject: str = 'Test Subject'

print(subjects.add(subject))
