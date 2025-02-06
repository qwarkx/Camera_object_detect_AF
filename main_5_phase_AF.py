import cv2
import numpy as np
import cvzone

from scipy.signal import correlate2d
'''
def shift_image(image, shift):
    shifted = np.zeros_like(image)
    shifted[:, shift:] = image[:, :-shift]
    return shifted

def phase_difference(image1, image2):
    corr = correlate2d(image1, image2, mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    return x - (image1.shape[1] // 2)

def autofocus_phase(image, max_shift=2):
    best_focus = None
    best_shift = 0
    for shift in range(1, max_shift):
        shifted_image = shift_image(image, shift)
        diff = phase_difference(image, shifted_image)
        if best_focus is None or abs(diff) < abs(best_focus):
            best_focus = diff
            best_shift = shift
    return best_shift


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Fázis-alapú autofókusz
    shift = autofocus_phase(gray)
    print(f"Optimal shift: {shift}")

    # Jelenítse meg a képet
'''

def simulate_phase_shift(image, shift_pixels=3):
    """Simulate a phase shift by shifting the image horizontally."""
    shifted_image = np.roll(image, shift_pixels, axis=1)
    return shifted_image


def compute_phase_difference(image1, image2):
    """Compute the phase difference between two images."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform FFT on both images
    f1 = np.fft.fft2(gray1)
    f2 = np.fft.fft2(gray2)

    # Compute cross-power spectrum
    cross_power = (f1 * f2.conjugate()) / np.abs(f1 * f2.conjugate())

    # Inverse FFT to get the phase difference
    phase_difference = np.fft.ifft2(cross_power).real

    # Normalize phase difference for visualization
    phase_diff_normalized = cv2.normalize(phase_difference, None, 0, 255, cv2.NORM_MINMAX)
    phase_diff_normalized = np.uint8(phase_diff_normalized)

    return phase_diff_normalized


def overlay_phase_difference(image, phase_diff, roi):
    """Overlay phase difference on the original image within the ROI."""
    phase_diff_bgr = cv2.cvtColor(phase_diff, cv2.COLOR_GRAY2BGR)

    # Copy the original image to overlay phase difference
    overlayed_image = image.copy()
    overlayed_image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = cv2.addWeighted(
        image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]], 0.5, phase_diff_bgr, 0.5, 0
    )

    return overlayed_image


def draw_rectangle_on_frame(frame, roi):
    """Draw a rectangle on the frame."""
    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 52)
    # cv2.rectangle(frame, (64,300), (191,300), (0,255,0), 2)
def phase_detection_with_overlay():
    # Open the camera
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Define a smaller rectangle in the center of the image
        height, width, _ = frame.shape
        roi_width, roi_height = width // 4, height // 4  # Rectangle size
        roi_x, roi_y = (width - roi_width) // 2, (height - roi_height) // 2
        roi = (roi_x, roi_y, roi_width, roi_height)

        # Extract ROI and simulate phase shift
        cropped = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        shifted_cropped = simulate_phase_shift(cropped)
        phase_diff = compute_phase_difference(cropped, shifted_cropped)

        # Overlay phase difference on the original frame
        frame_with_overlay = overlay_phase_difference(frame, phase_diff, roi)

        # Draw rectangle to highlight the ROI
        draw_rectangle_on_frame(frame_with_overlay, roi)

        # Display the frame with overlay
        cv2.imshow("Phase Detection with Overlay", frame_with_overlay)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()

# Run the function to see the rectangle
phase_detection_with_overlay()


'''
def phase_detection_with_overlay():
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Define a smaller rectangle in the center of the image
        height, width, _ = frame.shape
        roi_width, roi_height = width // 6, height // 6  # Smaller rectangle size for clarity
        roi_x, roi_y = width // 2 - roi_width // 2, height // 2 - roi_height // 2
        roi = (roi_x, roi_y, roi_width, roi_height)

        # Draw rectangle on the frame to visualize ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

        # Extract the region of interest (ROI)
        roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Simulate a phase shift within the ROI
        shifted_roi_frame = simulate_phase_shift(roi_frame, shift_pixels=5)

        # Compute phase difference only in the ROI
        phase_diff_roi = compute_phase_difference(roi_frame, shifted_roi_frame)

        # Overlay the phase difference on the original frame within the ROI
        overlayed_frame = overlay_phase_difference(frame, phase_diff_roi, roi)

        # Draw a smaller rectangle on the overlayed frame to visualize ROI
        cv2.rectangle(overlayed_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

        cvzone.cornerRect(overlayed_frame, (roi_x, roi_y, roi_width, roi_height))
        
        # Display the frame with overlay and rectangle
        cv2.imshow("Phase Detection with ROI Overlay", overlayed_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()
'''

# Run the phase detection with overlay function
phase_detection_with_overlay()