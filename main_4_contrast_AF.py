import cv2
import numpy as np

def measure_sharpness_and_overlay(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image and then the variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert laplacian result to 8-bit for visualization
    abs_laplacian = np.absolute(laplacian)
    sharpness_map = np.uint8(abs_laplacian)

    # Normalize sharpness map to display as overlay
    sharpness_overlay = cv2.normalize(sharpness_map, None, 0, 255, cv2.NORM_MINMAX)

    # Convert overlay to 3-channel BGR image
    sharpness_overlay_bgr = cv2.cvtColor(sharpness_overlay, cv2.COLOR_GRAY2BGR)

    # Blend the original image with the sharpness overlay using transparency
    alpha = 0.5  # Transparency factor (0: only original image, 1: only overlay)
    overlayed_image = cv2.addWeighted(image, 1 - alpha, sharpness_overlay_bgr, alpha, 0)

    return overlayed_image

def autofocus_with_overlay():
    # Open the camera
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera or the appropriate index

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Measure sharpness and get overlay
        overlayed_frame = measure_sharpness_and_overlay(frame)

        # Display the frame with overlay
        cv2.imshow("Autofocus with Contrast Overlay", overlayed_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()

# Run the autofocus with overlay function
autofocus_with_overlay()