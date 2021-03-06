package com.waicool20.djl.util

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.index.NDIndex
import ai.djl.translate.Transform

/**
 * Transforms top left xy coordinates to center xy coordinates
 * Input data array must be <category> <x> <y> <w> <h>
 */
class TopLeftXYToCenterXY : Transform {
    override fun transform(array: NDArray): NDArray {
        val x = array.get(NDIndex("..., 1"))
        val y = array.get(NDIndex("..., 2"))
        val w = array.get(NDIndex("..., 3"))
        val h = array.get(NDIndex("..., 4"))
        return array.apply {
            set(NDIndex("..., 1"), x + w / 2)
            set(NDIndex("..., 2"), y + h / 2)
        }
    }
}