import cv2


# average hash algorithm
def average_hash(img):
    # resize the image into 8*8
    img = cv2.resize(img, (8, 8))
    # transfer into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # the initial sum of pixels is 0
    s = 0
    # the initial value of hash is ''
    hash_str = ''
    # traverse to sum the pixels
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # the average of grayscale
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# compare the hash value of two images
def compare_hash(hash1, hash2):
    # count the different bits of two hash values(Hamming distance)
    # the smaller distance, the more similar of two images
    n = 0
    # ensure the same length of hash values
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


# get face recognition training data
face_cascade = cv2.CascadeClassifier(r"..\haarcascade_frontalface_default.xml")

# read and deal with the target image
t_img = cv2.imread("query2.jpg")
grayimg = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
# detect the human face in the target image
test_faces = face_cascade.detectMultiScale(
    grayimg,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)
if len(test_faces) > 0:
    for (x, y, w, h) in test_faces:
        resize_t_img = t_img[y:y + h, x:x + w]
else:
    print("please choose another image")
    quit()

# read the video
# input your own video address
vc = cv2.VideoCapture("../")

if vc.isOpened():
    # get the first frame
    is_capturing, frame = vc.read()
else:
    is_capturing = False

n = 0
flag = 0
while is_capturing:
    n = n + 1
    is_capturing, frame = vc.read()

    # every 25 frames(about 1 second)
    if n % 25 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # detect the human face in each frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                resize_frame = frame[y:y + h, x:x + w]

            # only compare the similarity of face regions
            hash1 = average_hash(resize_frame)
            hash2 = average_hash(resize_t_img)
            n3 = compare_hash(hash1, hash2)

            # set a threshold as 10 manually
            if n3 <= 10:
                time = n / 25
                print('The similarity：', n3)
                print("appear time: %d second of the video" % time)

                # frame the face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'target', (x, y - 7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                # represent the result
                cv2.imshow('target', frame)
                c = cv2.waitKey(0)
                # input the place you want to save the target frame
                cv2.imwrite(r"../" + 'humanface' + '_' + str(n) + '.jpg', frame)
                flag = 1
                break
            else:
                continue

        else:
            continue

    else:
        continue

if flag == 0:
    print("Could not find anything.")

vc.release()
