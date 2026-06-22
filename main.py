import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path="models/gesture_recognizer.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

recognizer = GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_count = 0

blur_active=False
blur_strength=0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    frame_count += 1

    result = recognizer.recognize_for_video(
        mp_image,
        frame_count * 33
    )

    if result.gestures:

        gesture_name=(
            result.gestures[0][0]
            .category_name
        )

        if gesture_name=="Victory":
            blur_active=True
        else:
            blur_active=False

    else:
        blur_active=False

    if blur_strength > 0:

        kernel=int(blur_strength)

        if kernel % 2 == 0:
            kernel += 1

        frame=cv2.GaussianBlur(
            frame,
            (kernel, kernel),
            0
        )

    if blur_active:
        blur_strength=min(35, blur_strength + 2)

    else:
        blur_strength=max(0, blur_strength-2)

    cv2.imshow(
        "Gesture Detection",
        frame
    )

    if cv2.getWindowProperty(
        "Gesture Detection",
        cv2.WND_PROP_VISIBLE
    ) <1:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()