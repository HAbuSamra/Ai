import cv2 as cv  

def detection(frame, face_detection, eye_detection):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # استخراج منطقة الوجه
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # كشف العيون داخل الوجه
        eyes = eye_detection.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame


def move_bodys(frame, MoveBody):
    move = MoveBody.apply(frame)
    contours, _ = cv.findContours(move, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    for contour in contours:
        if cv.contourArea(contour) > 500:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def main():
    # استخدام كاميرا واحدة
    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        print("❌ خطأ: لم يتم فتح الكاميرا بشكل صحيح!")
        return

    face_detection = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_detection = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
    MoveBody = cv.createBackgroundSubtractorMOG2()

    # التحقق من نجاح تحميل ملفات الكشف
    if face_detection.empty() or eye_detection.empty():
        print("❌ خطأ: لم يتم تحميل ملفات Haar Cascade بشكل صحيح!")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # دمج الكشفين في نفس الإطار
        frame_with_faces = detection(frame, face_detection, eye_detection)
        frame_with_movement = move_bodys(frame_with_faces, MoveBody)

        # عرض الصورة مع الرسومات
        cv.imshow('Face, Eye and Body Detection', frame_with_movement)

        # الضغط على "q" للخروج
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # إغلاق الكاميرا والنوافذ
    camera.release()
    cv.destroyAllWindows()

# تشغيل البرنامج
main()
