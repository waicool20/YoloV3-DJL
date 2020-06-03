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
        val xMin = array.get(NDIndex("..., 1"))
        val yMin = array.get(NDIndex("..., 2"))
        val xMax = array.get(NDIndex("..., 3"))
        val yMax = array.get(NDIndex("..., 4"))
        return array.apply {
            set(NDIndex("..., 3"), xMax - xMin)
            set(NDIndex("..., 4"), yMax - yMin)
        }
    }
}