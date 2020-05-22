package com.waicool20.djl.util

import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.index.NDIndex
import kotlin.math.max
import kotlin.math.min

object YoloUtils {
    fun whIOU(iwh1: NDArray, iwh2: NDArray): NDArray {
        val wh1 = iwh1.expandDims(1)
        val wh2 = iwh2.expandDims(0)
        val inter = NDArrays.minimum(wh1, wh2).prod(intArrayOf(2))
        return inter / (wh1.prod(intArrayOf(2)) + wh2.prod(intArrayOf(2)) - inter)
    }

    fun bboxIOUs(boxes1: NDArray, boxes2: NDArray): NDArray {
        val b1w = boxes1.get(NDIndex(":, 2")) / 2
        val b1h = boxes1.get(NDIndex(":, 3")) / 2

        val b1x1 = boxes1.get(NDIndex(":, 0")) - b1w
        val b1x2 = boxes1.get(NDIndex(":, 0")) + b1w
        val b1y1 = boxes1.get(NDIndex(":, 1")) - b1h
        val b1y2 = boxes1.get(NDIndex(":, 1")) + b1h

        val b2w = boxes2.get(NDIndex(":, 2")) / 2
        val b2h = boxes2.get(NDIndex(":, 3")) / 2

        val b2x1 = boxes2.get(NDIndex(":, 0")) - b2w
        val b2x2 = boxes2.get(NDIndex(":, 0")) + b2w
        val b2y1 = boxes2.get(NDIndex(":, 1")) - b2h
        val b2y2 = boxes2.get(NDIndex(":, 1")) + b2h

        val rx1 = NDArrays.maximum(b1x1, b2x1)
        val ry1 = NDArrays.maximum(b1y1, b2y1)
        val rx2 = NDArrays.minimum(b1x2, b2x2)
        val ry2 = NDArrays.minimum(b1y2, b2y2)

        val interArea = (rx1 - rx2) * (ry1 - ry2)

        val b1a = (b1x2 - b1x1) * (b1y2 - b1y1)
        val b2a = (b2x2 - b2x1) * (b2y2 - b2y1)
        return (interArea / (b1a + b2a - interArea + 1e-16f)).clip(0, 1)
    }

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

        val interArea = (rx1 - rx2) * (ry1 - ry2)

        val b1a = (b1x2 - b1x1) * (b1y2 - b1y1)
        val b2a = (b2x2 - b2x1) * (b2y2 - b2y1)
        return (interArea / (b1a + b2a - interArea + 1e-16f)).coerceIn(0f, 1f)
    }

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
}