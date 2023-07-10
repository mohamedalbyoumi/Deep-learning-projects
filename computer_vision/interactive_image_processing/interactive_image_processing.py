import cv2
import numpy as np
import datetime

class ImageProcessingGUI:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.original_image = np.copy(self.image)
        self.roi = None
        self.history = []
        self.most_color = None
        self.most_color_invr = None
        self.trackbar_width = 100
        self.trackbar_height = 10
        self.trackbar_x = 5
        self.trackbar_y = 5

        self.cv2_named_window = "Image"
        cv2.namedWindow(self.cv2_named_window)
        cv2.setMouseCallback(self.cv2_named_window, self.mouse_callback)

        self.create_gui()

    def create_gui(self):
        cv2.createTrackbar("Brightness", self.cv2_named_window, 0, 255, self.adjust_brightness)
        cv2.createTrackbar("Contrast", self.cv2_named_window, 0, 20, self.adjust_contrast)
        cv2.createTrackbar("Saturation", self.cv2_named_window, 0, 20, self.adjust_saturation)

        # Adjust the position of the image display area
        image_x = self.trackbar_x
        image_y = self.trackbar_y + self.trackbar_height + 10
        image_width = self.image.shape[1]
        image_height = self.image.shape[0]

        # Resize the window to fit the image and trackbars
        window_width = image_x + image_width + 10
        window_height = image_y + image_height + 10
        cv2.resizeWindow(self.cv2_named_window, window_width, window_height)


    def resize_image(self, width, height):
        self.image = cv2.resize(self.image, (width, height))

    def display_image(self):
        window_size = cv2.getWindowImageRect(self.cv2_named_window)
        image_width = window_size[2] - self.trackbar_x - 10
        image_height = window_size[3] - self.trackbar_y - self.trackbar_height - 10

        # Resize the image to fit the available space
        self.resize_image(image_width, image_height)


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.history.append(np.copy(self.image))

        if event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(self.image, (x, y), (x+500, y+500), (0, 255, 0), 2)
            self.roi = self.image[y:y+500, x:x+500]

        if event == cv2.EVENT_RBUTTONDOWN:
            self.history.append(np.copy(self.image))

        if event == cv2.EVENT_RBUTTONUP:
            cv2.circle(self.image, (x, y), 50, (0, 0, 255), 2)

        if event == cv2.EVENT_MBUTTONDOWN:
            self.history.append(np.copy(self.image))

        if event == cv2.EVENT_MBUTTONUP:
            translation_matrix = np.float32([[1, 0, x - 100], [0, 1, y - 100]])
            self.image = cv2.warpAffine(self.image, translation_matrix, (self.image.shape[1], self.image.shape[0]))

    def adjust_brightness(self, value):
        self.image = cv2.add(np.array([value]), self.image)

    def adjust_contrast(self, value):
        self.image = cv2.multiply(self.image, value / 10.0)

    def adjust_saturation(self, value):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * (value / 10.0)
        self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def count_most_color(self):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Calculate the color histogram
        hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # Flatten the histogram to a 1D array
        hist_flat = hist.flatten()

        # Get the index of the bin with the maximum frequency count
        max_bin_index = np.argmax(hist_flat)

        # Convert the bin index to corresponding hue and saturation values
        hue_bin = max_bin_index % 180
        saturation_bin = max_bin_index // 180

        # Calculate the maximum hue and saturation values
        max_hue = hue_bin * (180 / 256)
        max_saturation = saturation_bin * (256 / 256)

        # Create a 3-channel HSV image with the maximum hue, saturation, and value
        max_color_hsv = np.zeros_like(self.image)
        max_color_hsv[..., 0] = max_hue
        max_color_hsv[..., 1] = max_saturation
        max_color_hsv[..., 2] = 255

        # Convert the HSV image to RGB
        max_color_rgb = cv2.cvtColor(max_color_hsv, cv2.COLOR_HSV2BGR)

        # Retrieve the pixel value at the corresponding location
        max_color = max_color_rgb[0, 0]

        # Set the rectangle and text color based on the most occurring color
        self.most_color = tuple(max_color.tolist())

    def count_most_color_invr(self):
         most_color = self.most_color
         self.most_color_invr = tuple(255 - value for value in most_color)       

    def display_time(self):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # Set the font, scale, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2

        # Get the text size
        text_size, _ = cv2.getTextSize(current_time, font, scale, thickness)

        # Calculate the coordinates for the rectangle
        rect_width = text_size[0] + 20
        rect_height = text_size[1] + 20
        rect_x = 5
        rect_y = 5

        # Erase the previous printed text by filling a rectangle
        cv2.rectangle(self.image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), self.most_color, -1)

        # Add the current time text
        cv2.putText(self.image, current_time, (rect_x + 10, rect_y + rect_height - 10), font, scale, self.most_color_invr, thickness, cv2.LINE_AA)
        
    def display_help(self):
        help_text = """
        Mouse Events:
            - Left Button Drag: Draw Rectangle
            - Right Button Drag: Draw Circle
            - Middle Button Drag: Image Translation
        Keyboard Shortcuts:
            - 'g': Convert to Grayscale
            - 'r': Reset to Original Image
            - 's': Save Image
            - 'c': Crop Region of Interest
            - 'z': Undo Previous Operation
            - 'q': Quit
            - 'h': Display Help
        """

        # Set the font, scale, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        # Get the text size
        text_size, _ = cv2.getTextSize(help_text, font, scale, thickness)

        # Calculate the size of the help window
        window_width = text_size[0] + 20
        window_height = text_size[1] + 40

        # Calculate the coordinates for the text
        text_x = 10
        text_y = 20

        # Create a black bar for the help text display
        bar_height = 250
        bar_width = 440
        bar = np.zeros((bar_height, bar_width, 3), np.uint8)
        bar[bar == 0] = 255

        # Add the help text to the bar
        lines = help_text.strip().split("\n")
        lines[4] = lines[4].strip()
        for i, line in enumerate(lines):
            cv2.putText(bar, line, (text_x, text_y + i * 20), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

        # Display the bar in a separate window
        cv2.imshow("Help Display", bar)
        


    def undo_operation(self):
        if self.history:
            self.image = self.history.pop()

    def reset_image(self):
        self.image = np.copy(self.original_image)
        self.roi = None
        self.history = []

    def convert_to_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def save_image(self):
        cv2.imwrite("processed_image.jpg", self.image)
        print("Image saved successfully.")

    def crop_roi(self):
        if self.roi is not None:
            self.image = np.copy(self.roi)
            self.roi = None
            self.history = []

    def main(self):
        while True:
            self.display_time()

            cv2.imshow(self.cv2_named_window, self.image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or cv2.getWindowProperty(self.cv2_named_window, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("h"):
                self.display_help()
            elif key == ord("z"):
                self.undo_operation()
            elif key == ord("r"):
                self.reset_image()
            elif key == ord("g"):
                self.convert_to_grayscale()
            elif key == ord("s"):
                self.save_image()
            elif key == ord("c"):
                self.crop_roi()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ")
    gui = ImageProcessingGUI(image_path)
    gui.count_most_color()
    gui.count_most_color_invr()
    gui.display_image()  # Initial resizing
    gui.main()
