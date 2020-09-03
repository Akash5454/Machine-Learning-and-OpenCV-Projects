import sys
import face_recognition


def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    known_image = face_recognition.load_image_file(imageA)
    unknown_image = face_recognition.load_image_file(imageB)


    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    return results

if __name__ == "__main__":
  output = compare_images(sys.argv[1], sys.argv[2])
  print(output)