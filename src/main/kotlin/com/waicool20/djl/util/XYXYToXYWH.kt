package com.waicool20.djl.util

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.index.NDIndex
import ai.djl.translate.Transform

/**
 * Transform labels from Xmin Ymin Xmax Ymax to top left corner based x y w h format
 * Input data array must be <category> <Xmin> <Ymin> <Xmax> <Ymax>
 */
class XYXYToXYWH : Transform {
    override fun transform(array: NDArray): NDArray {
        val xMin = array.get(NDIndex(array.fEllipsis() + 1))
        val yMin = array.get(NDIndex(array.fEllipsis() + 2))
        val xMax = array.get(NDIndex(array.fEllipsis() + 3))
        val yMax = array.get(NDIndex(array.fEllipsis() + 4))
        return array.apply {
            set(NDIndex(array.fEllipsis() + 3), xMax - xMin)
            set(NDIndex(array.fEllipsis() + 4), yMax - yMin)
        }
    }
}