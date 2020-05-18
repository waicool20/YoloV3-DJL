package com.waicool20.djl.util

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import kotlin.math.max

object YoloUtils {
    fun whIOU(wh1: NDArray, wh2: NDArray): NDArray {
        val wh2t = wh2.transpose()
        val w1 = wh1[0]
        val h1 = wh1[1]
        val w2 = wh2t[0]
        val h2 = wh2t[1]

        val inter = NDArrays.minimum(w1, w2) * NDArrays.minimum(h1, h2)
        val union = (w1 * h1 + w2 * h2 + 1e-16) - inter
        return inter / union
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
        val rx2 = max(b1x2, b2x2)
        val ry2 = max(b1y2, b2y2)

        val interArea = (rx1 - rx2) * (ry1 - ry2)

        val b1a = (b1x2 - b1x1) * (b1y2 - b1y1)
        val b2a = (b2x2 - b2x1) * (b2y2 - b2y1)
        return interArea / (b1a + b2a - interArea + 1e-16f)
    }
}