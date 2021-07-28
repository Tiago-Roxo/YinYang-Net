# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
@author: Tiago Roxo, UBI
@date: 2021
"""

import numpy as np
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.head_detect_config import KEYPOINT_THRESHOLD, COCO_PERSON_KEYPOINT_NAMES, KEYPOINT_CONNECTION_RULES


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

class Visualizer:
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.
    To obtain a consistent style, implement custom drawing functions with the primitive
    methods instead.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=0):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): image metadata.
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode


    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        # Allow drawing of various predictions in the same image
        #for prediction in predictions:
        #    self.draw_and_connect_keypoints(prediction)

        for kp in predictions:
           self.draw_and_connect_keypoints(kp)

        return self.output

    def draw_bb(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        # Allow drawing of various predictions in the same image
        #for prediction in predictions:
        #    self.draw_and_connect_keypoints(prediction)

        for bb in predictions:
            self.draw_box(bb)

        return self.output

    def draw_head(self, predictions):
        
        if (len(predictions)==0):
            return None, None

        self.output, bbox = self.draw_head_kp(predictions[0])
        # for head_kp in predictions:
        #     self.draw_head_kp(head_kp)

        return self.output, bbox



    def draw_and_connect_keypoints(self, keypoints):
        """
        Draws keypoints of an instance and follows the rules for keypoint connections
        to draw lines between appropriate keypoints. This follows color heuristics for
        line color.

        Args:
            keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
                and the last dimension corresponds to (x, y, probability).

        Returns:
            output (VisImage): image object with visualizations.
        """
        visible = {}
        keypoint_names = COCO_PERSON_KEYPOINT_NAMES
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > KEYPOINT_THRESHOLD:
                self.draw_circle((x, y), color=RED)
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)

        if KEYPOINT_CONNECTION_RULES:
            for kp0, kp1, color in KEYPOINT_CONNECTION_RULES:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = tuple(x / 255.0 for x in color)
                    self.draw_line([x0, x1], [y0, y1], color=color)

        # draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
        # Note that this strategy is specific to person keypoints.
        # For other keypoints, it should just do nothing
        try:
            ls_x, ls_y = visible["left_shoulder"]
            rs_x, rs_y = visible["right_shoulder"]
            mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
        except KeyError:
            pass
        else:
            # draw line from nose to mid-shoulder
            nose_x, nose_y = visible.get("nose", (None, None))
            if nose_x is not None:
                self.draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=RED)

            try:
                # draw line from mid-shoulder to mid-hip
                lh_x, lh_y = visible["left_hip"]
                rh_x, rh_y = visible["right_hip"]
            except KeyError:
                pass
            else:
                mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
                self.draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=RED)
        return self.output

    def draw_head_kp(self, keypoints):
        """
        Draws keypoints of an instance and follows the rules for keypoint connections
        to draw lines between appropriate keypoints. This follows color heuristics for
        line color.

        Args:
            keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
                and the last dimension corresponds to (x, y, probability).

        Returns:
            output (VisImage): image object with visualizations.
        """
        visible = {}
        keypoint_names = COCO_PERSON_KEYPOINT_NAMES
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            keypoint_name = keypoint_names[idx]
            if prob > KEYPOINT_THRESHOLD:
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)


        draw_bbox_top_left, draw_bbox_bot_right = self.new_approach(visible)

        if draw_bbox_top_left == None:
            print("Missing KP for head drawing:")
            return None, None

        # Top left corner coordinates box and bottom right coordinates, for this order
        draw_bbox_top_left_x, draw_bbox_top_left_y = draw_bbox_top_left
        draw_bbox_bot_right_x, draw_bbox_bot_right_y = draw_bbox_bot_right

        extra_margin = 2

        # Adaptation to COCO GT
        width  = draw_bbox_bot_right_x - draw_bbox_top_left_x
        height = draw_bbox_bot_right_y - draw_bbox_top_left_y  
        
        self.draw_box([draw_bbox_top_left_x-extra_margin, draw_bbox_top_left_y-extra_margin, 
            draw_bbox_bot_right_x+extra_margin, draw_bbox_bot_right_y+extra_margin])

        bbox = [draw_bbox_top_left_x-extra_margin, draw_bbox_top_left_y-extra_margin, 
            draw_bbox_top_left_x+width+extra_margin, draw_bbox_top_left_y+height+extra_margin]

        return self.output, bbox
    

    def new_approach(self, visible):

        try:
            left_shoulder_x, left_shoulder_y   = visible["left_shoulder"]
            right_shoulder_x, right_shoulder_y = visible["right_shoulder"]

            # Rightmost shoulder in relation to the top left corner of image
            rightmost_shoulder = max(right_shoulder_x, left_shoulder_x)
            facing_front = True
            if rightmost_shoulder != left_shoulder_x:
                facing_front = False

            h, _, _ = self.img.shape
            head_body_prop = h/9

            right_x, right_y = visible["right_ear"]
            left_x, left_y = visible["left_ear"]
            ref_x = abs(right_x+left_x)/2
            ref_y = abs(right_y+left_y)/2

            draw_bbox_top_left  = (ref_x-head_body_prop, ref_y-head_body_prop)
            draw_bbox_bot_right = (ref_x+head_body_prop, ref_y+head_body_prop)

            return draw_bbox_top_left, draw_bbox_bot_right
        
        except:
            return None, None

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output
    
    def draw_circle(self, circle_coord, color, radius=3):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="r", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        # Adaptation to COCO GT
        width  = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )

        return self.output