package com.waicool20.djl.util

import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import kotlin.math.max
import kotlin.math.min

object YoloUtils {
    enum class IOUType {
        /**
         * Standard Intersection over Union
         */
        IOU,

        /**
         * Generalized Intersection over Union
         */
        GIOU,

        /**
         * Distance Intersection over Union
         */
        DIOU,

        /**
         * Complete Intersection over Union
         */
        CIOU
    }

    /**
     * This calculates the iou between two boxes that share the same center point
     * hence the need for width and height only.
     *
     * @param iwh1 Box 1 width and height (bs, m, 2)
     * @param iwh2 Box 2 width and height (bs, n, 2)
     *
     * @return IOUs (bs, m*n)
     */
    fun whIOU(iwh1: NDArray, iwh2: NDArray): NDArray {
        val wh1 = iwh1.expandDims(1)
        val wh2 = iwh2.expandDims(0)
        val inter = NDArrays.minimum(wh1, wh2).prod(intArrayOf(2))
        return inter / (wh1.prod(intArrayOf(2)) + wh2.prod(intArrayOf(2)) - inter + 1e-16)
    }

    /**
     * Calculates the corresponding IOU between boxes
     *
     * @param boxes1 Box 1 center based XYWH (bs, 4)
     * @param boxes2 Box 2 center based XYWH (bs, 4)
     * @param type Type of IOU to calculate
     *
     * @return IOUs (bs)
     */
    fun bboxIOUs(boxes1: NDArray, boxes2: NDArray, type: IOUType = IOUType.IOU): NDArray {
        val (b1x1, b1x2, b1y1, b1y2) = centerXYWHToXYXY(boxes1)
        val (b2x1, b2x2, b2y1, b2y2) = centerXYWHToXYXY(boxes2)

        val inter = run {
            val rx1 = NDArrays.maximum(b1x1, b2x1)
            val rx2 = NDArrays.minimum(b1x2, b2x2)
            val ry1 = NDArrays.maximum(b1y1, b2y1)
            val ry2 = NDArrays.minimum(b1y2, b2y2)
            (rx2 - rx1).clip(0, 1) * (ry2 - ry1).clip(0, 1)
        }

        val w1 = b1x2 - b1x1
        val h1 = b1y2 - b1y1
        val w2 = b2x2 - b2x1
        val h2 = b2y2 - b2y1

        val union = w1 * h1 + w2 * h2 - inter + 1e-16

        val iou = inter / union
        if (type == IOUType.IOU) return iou

        val cw = NDArrays.maximum(b1x2, b2x2) - NDArrays.minimum(b1x1, b2x1)
        val ch = NDArrays.maximum(b1y2, b2y2) - NDArrays.minimum(b1y1, b2y1)

        if (type == IOUType.GIOU) {
            val ca = cw * ch + 1e-16
            return iou - ((ca - union) / ca)
        }

        val c2 = cw.square() + ch.square() + 1e-16
        val rho2 = (((b2x1 + b2x2) - (b1x1 + b1x2)).square() + ((b2y1 + b2y2) - (b1y1 + b1y2)).square()) / 4
        val diou = iou - (rho2 / c2)
        if (type == IOUType.DIOU) return diou
        if (type == IOUType.CIOU) {
            val c = 0.405284735 // 4 / pi^2
            val v = (((w2 / h2).atan() - (w1 / h1).atan()).square() * c + 1e-16).gradDetach()
            val a = (v / ((iou.neg() + 1) + v)).gradDetach()
            return diou - a * v
        }
        error("Invalid IOU type")
    }

    /**
     * Calculates the standard IOU between boxes
     *
     * @param box1 Box 1 center based XYWH (4)
     * @param box2 Box 2 center based XYWH (4)
     *
     * @return IOU as Float
     */
    fun bboxIOU(box1: NDArray, box2: NDArray): Float {
        val b1x1 = box1.getFloat(0) - box1.getFloat(2) / 2
        val b1x2 = box1.getFloat(0) + box1.getFloat(2) / 2
        val b1y1 = box1.getFloat(1) - box1.getFloat(3) / 2
        val b1y2 = box1.getFloat(1) + box1.getFloat(3) / 2

        val b2x1 = box2.getFloat(0) - box2.getFloat(2) / 2
        val b2x2 = box2.getFloat(0) + box2.getFloat(2) / 2
        val b2y1 = box2.getFloat(1) - box2.getFloat(3) / 2
        val b2y2 = box2.getFloat(1) + box2.getFloat(3) / 2

        val rx1 = max(b1x1, b2x1)
        val ry1 = max(b1y1, b2y1)
        val rx2 = min(b1x2, b2x2)
        val ry2 = min(b1y2, b2y2)

        val interArea = (rx2 - rx1).coerceAtLeast(0f) * (ry2 - ry1).coerceAtLeast(0f)

        val b1a = (b1x2 - b1x1) * (b1y2 - b1y1)
        val b2a = (b2x2 - b2x1) * (b2y2 - b2y1)
        return (interArea / (b1a + b2a - interArea + 1e-16f)).coerceIn(0f, 1f)
    }

    /**
     * Applies non-maximum suppression with a given IoU threshold to a list of [DetectedObjects]
     *
     * @param iouThreshold IoU Threshold (0.0 - 1.0)
     * @param boxes Detection results
     *
     * @return Detection results with NMS applied
     */
    fun nms(iouThreshold: Double, boxes: List<DetectedObjects.DetectedObject>): List<DetectedObjects.DetectedObject> {
        val input = boxes.toMutableList()
        val output = mutableListOf<DetectedObjects.DetectedObject>()
        while (input.isNotEmpty()) {
            val best = input.maxBy { it.probability } ?: continue
            input.remove(best)
            input.removeAll { it.className == best.className && it.boundingBox.getIoU(best.boundingBox) >= iouThreshold }
            output.add(best)
        }
        return output
    }

    private fun centerXYWHToXYXY(box: NDArray): NDList {
        val w = box.get(NDIndex(":, 2")) / 2
        val h = box.get(NDIndex(":, 3")) / 2

        val x1 = box.get(NDIndex(":, 0")) - w
        val x2 = box.get(NDIndex(":, 0")) + w
        val y1 = box.get(NDIndex(":, 1")) - h
        val y2 = box.get(NDIndex(":, 1")) + h
        return NDList(x1, x2, y1, y2)
    }
}