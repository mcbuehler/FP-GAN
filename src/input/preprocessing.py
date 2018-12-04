import numpy as np
import cv2 as cv
import tensorflow as tf
from util.log import get_logger

logger = get_logger()


class Preprocessor:

    @staticmethod
    def equalize(image):  # Proper colour image intensity equalization
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        output = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        return output


class UnityPreprocessor(Preprocessor):
    def __init__(self,
                 testing=False,
                 eye_image_shape=(36, 60)):

        """Create queues and threads to read and preprocess data."""
        self._short_name = 'UnityEyes'
        self.do_augmentation = not testing
        if testing:
            self._short_name += ':test'

        # Cache some parameters
        self._eye_image_shape = eye_image_shape

        # Define bounds for noise values for different augmentation types
        self._difficulty = 0.0
        self._augmentation_ranges = {  # (easy, hard)
            'translation': (2.0, 10.0),
            'rotation': (0.1, 2.0),
            'intensity': (0.5, 20.0),
            'blur': (0.1, 1.0),
            'scale': (0.01, 0.1),
            'rescale': (1.0, 0.2),
            'num_line': (0.0, 2.0),
            'heatmap_sigma': (5.0, 2.5),
        }

    def augment(self, eye):
        oh, ow = self._eye_image_shape

        # Draw line randomly
        num_line_noise = int(np.round(self._noisy_value_from_type('num_line')))
        if num_line_noise > 0:
            line_rand_nums = np.random.rand(5 * num_line_noise)
            for i in range(num_line_noise):
                j = 5 * i
                lx0, ly0 = int(ow * line_rand_nums[j]), oh
                lx1, ly1 = ow, int(oh * line_rand_nums[j + 1])
                direction = line_rand_nums[j + 2]
                if direction < 0.25:
                    lx1 = ly0 = 0
                elif direction < 0.5:
                    lx1 = 0
                elif direction < 0.75:
                    ly0 = 0
                line_colour = int(255 * line_rand_nums[j + 3])
                eye = cv.line(eye, (lx0, ly0), (lx1, ly1),
                              color=(line_colour, line_colour, line_colour),
                              thickness=int(6 * line_rand_nums[j + 4]),
                              lineType=cv.LINE_AA)

        # Rescale image if required
        rescale_max = self._value_from_type('rescale')
        if rescale_max < 1.0:
            rescale_noise = np.random.uniform(low=rescale_max, high=1.0)
            interpolation = cv.INTER_CUBIC
            eye = cv.resize(eye, dsize=(0, 0), fx=rescale_noise,
                            fy=rescale_noise,
                            interpolation=interpolation)
            eye = cv.equalizeHist(eye)
            eye = cv.resize(eye, dsize=(ow, oh), interpolation=interpolation)

        # Add rgb noise to eye image
        intensity_noise = int(self._value_from_type('intensity'))
        if intensity_noise > 0:
            eye = eye.astype(np.int16)
            eye += np.random.randint(low=-intensity_noise,
                                     high=intensity_noise,
                                     size=eye.shape, dtype=np.int16)
            cv.normalize(eye, eye, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            eye = eye.astype(np.uint8)

        # Add blur to eye image
        blur_noise = self._noisy_value_from_type('blur')
        if blur_noise > 0:
            eye = cv.GaussianBlur(eye, (7, 7), 0.5 + np.abs(blur_noise))
        return eye

    def preprocess(self, full_image, json_data):
        """Use annotations to segment eyes and calculate gaze direction."""
        result_dict = dict()

        ih, iw = int(full_image.shape[0]), int(full_image.shape[1])  # image might have 2 or 3 channels
        iw_2, ih_2 = 0.5 * int(iw), 0.5 * int(ih)
        oh, ow = self._eye_image_shape

        def process_coords(coords_list):
            coords = [eval(l) for l in coords_list]
            return np.array([(x, ih - y, z) for (x, y, z) in coords])

        interior_landmarks = process_coords(json_data['interior_margin_2d'])
        caruncle_landmarks = process_coords(json_data['caruncle_2d'])
        iris_landmarks = process_coords(json_data['iris_2d'])

        # Only select almost frontal images
        h_pitch, h_yaw, _ = eval(json_data['head_pose'])
        if h_pitch > 180.0:  # Need to correct pitch
            h_pitch -= 360.0
        h_yaw -= 180.0  # Need to correct yaw
        # if abs(h_pitch) > 20 or abs(h_yaw) > 20:
        #     return None
        h_pitch = -h_pitch
        h_yaw = -h_yaw
        result_dict['head'] = np.asarray([np.radians(h_pitch), np.radians(h_yaw)],
                                   dtype=np.float32)

        # Prepare to segment eye image
        left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
        right_corner = interior_landmarks[8, :2]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                              np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

        # Centre axes to eyeball centre
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-iw_2], [-ih_2]]

        # Rotate eye image if requested
        rotate_mat = np.asmatrix(np.eye(3))
        rotation_noise = self._noisy_value_from_type('rotation')
        if rotation_noise > 0:
            rotate_angle = np.radians(rotation_noise)
            cos_rotate = np.cos(rotate_angle)
            sin_rotate = np.sin(rotate_angle)
            rotate_mat[0, 0] = cos_rotate
            rotate_mat[0, 1] = -sin_rotate
            rotate_mat[1, 0] = sin_rotate
            rotate_mat[1, 1] = cos_rotate

        # Scale image to fit output dimensions (with a little bit of noise)
        scale_mat = np.asmatrix(np.eye(3))
        scale = 1. + self._noisy_value_from_type('scale')
        scale_inv = 1. / scale
        np.fill_diagonal(scale_mat, ow / eye_width * scale)
        original_eyeball_radius = 71.7593
        eyeball_radius = original_eyeball_radius * scale_mat[
            0, 0]  # See: https://goo.gl/ZnXgDE
        result_dict['radius'] = np.float32(eyeball_radius)

        # Re-centre eye image such that eye fits (based on determined `eye_middle`)
        recentre_mat = np.asmatrix(np.eye(3))
        recentre_mat[0, 2] = iw / 2 - eye_middle[0] + 0.5 * eye_width * scale_inv
        recentre_mat[1, 2] = ih / 2 - eye_middle[
            1] + 0.5 * oh / ow * eye_width * scale_inv
        recentre_mat[0, 2] += self._noisy_value_from_type('translation')  # x
        recentre_mat[1, 2] += self._noisy_value_from_type('translation')  # y

        # Apply transforms
        transform_mat = recentre_mat * scale_mat * rotate_mat * translate_mat
        eye = cv.warpAffine(full_image, transform_mat[:2, :3], (ow, oh))

        # Store "clean" eye image before adding noises
        clean_eye = np.copy(eye)
        clean_eye = self.equalize(clean_eye)
        clean_eye = clean_eye.astype(np.float32)
        clean_eye *= 2.0 / 255.0
        clean_eye -= 1.0

        result_dict['clean_eye'] = clean_eye

        # Convert look vector to gaze direction in polar angles
        look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3]
        look_vec[0] = -look_vec[0]

        import util.gaze as gaze
        original_gaze = gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
        look_vec = rotate_mat * look_vec.reshape(3, 1)

        gaze = gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
        if gaze[1] > 0.0:
            gaze[1] = np.pi - gaze[1]
        elif gaze[1] < 0.0:
            gaze[1] = -(np.pi + gaze[1])
        result_dict['gaze'] = gaze.astype(np.float32)

        # Start augmentation
        if self.do_augmentation:
            eye = self.augment(eye)

        # Histogram equalization and preprocessing for NN
        eye = self.equalize(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0

        result_dict['eye'] = eye

        # Select and transform landmark coordinates
        iris_centre = np.asarray([
            iw_2 + original_eyeball_radius * -np.cos(original_gaze[0]) * np.sin(
                original_gaze[1]),
            ih_2 + original_eyeball_radius * -np.sin(original_gaze[0]),
        ])
        landmarks = np.concatenate([interior_landmarks[::2, :2],  # 8
                                    iris_landmarks[::4, :2],  # 8
                                    iris_centre.reshape((1, 2)),
                                    [[iw_2, ih_2]],  # Eyeball centre
                                    ])  # 18 in total
        landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant',
                                       constant_values=1))
        landmarks = np.asarray(landmarks * transform_mat.T)
        landmarks = landmarks[:, :2]  # We only need x, y
        result_dict['landmarks'] = landmarks.astype(np.float32)

        return_keys = ['clean_eye', 'eye', 'gaze']
        return [result_dict[k] for k in return_keys]

    def _value_from_type(self, augmentation_type):
        # Scale to be in range
        easy_value, hard_value = self._augmentation_ranges[augmentation_type]
        value = (hard_value - easy_value) * self._difficulty + easy_value
        value = (np.clip(value, easy_value, hard_value)
                 if easy_value < hard_value
                 else np.clip(value, hard_value, easy_value))
        return value

    def _noisy_value_from_type(self, augmentation_type):
        random_multipliers = []
        # Get normal distributed random value
        if len(random_multipliers) == 0:
            random_multipliers.extend(
                list(np.random.normal(size=(len(self._augmentation_ranges),))))
        return random_multipliers.pop() * self._value_from_type(augmentation_type)


class MPIIPreprocessor(Preprocessor):
    def __init__(self,
                 testing=False,
                 eye_image_shape=(36, 60)):
        self._eye_image_shape = eye_image_shape
        self.testing = testing

    def preprocess(self, image):
        # BGR to RGB conversion
        image = image[..., ::-1]

        if self._eye_image_shape is not None:
            oh, ow = self._eye_image_shape
            image = cv.resize(image, (ow, oh))
        image = self.equalize(image)
        image = image.astype(np.float32)
        image *= 2.0 / 255.0
        image -= 1.0
        return image