package com.waicool20.djl.util

import ai.djl.mxnet.engine.MxNDArray
import ai.djl.mxnet.jna.JnaUtils
import ai.djl.mxnet.jna.MxnetLibrary
import ai.djl.ndarray.NDArray
import com.sun.jna.ptr.PointerByReference
import kotlin.reflect.full.staticProperties
import kotlin.reflect.jvm.isAccessible

operator fun NDArray.plus(array: NDArray): NDArray = add(array)
operator fun NDArray.plus(number: Number): NDArray = add(number)
operator fun NDArray.minus(array: NDArray): NDArray = sub(array)
operator fun NDArray.minus(number: Number): NDArray = sub(number)
operator fun NDArray.times(array: NDArray): NDArray = mul(array)
operator fun NDArray.times(number: Number): NDArray = mul(number)

operator fun NDArray.unaryMinus(): NDArray = neg()

operator fun Number.plus(array: NDArray): NDArray = array + this
operator fun Number.minus(array: NDArray): NDArray = array.neg() + this
operator fun Number.times(array: NDArray): NDArray = array * this
operator fun Number.div(array: NDArray): NDArray = array.pow(-1) * this

operator fun NDArray.plusAssign(array: NDArray) {
    addi(array)
}

operator fun NDArray.plusAssign(number: Number) {
    addi(number)
}

operator fun NDArray.minusAssign(array: NDArray) {
    subi(array)
}

operator fun NDArray.minusAssign(number: Number) {
    subi(number)
}

operator fun NDArray.timesAssign(array: NDArray) {
    muli(array)
}

operator fun NDArray.timesAssign(number: Number) {
    muli(number)
}

operator fun NDArray.divAssign(array: NDArray) {
    divi(array)
}

operator fun NDArray.divAssign(number: Number) {
    divi(number)
}

private val MXLIB by lazy {
    JnaUtils::class.staticProperties.find { it.name == "LIB" }
        ?.apply { isAccessible = true }
        ?.invoke() as? MxnetLibrary
}

fun NDArray.gradDetach(): NDArray {
    val p = (this as MxNDArray).handle
    MXLIB?.MXNDArrayDetach(p, PointerByReference(p))
    return this
}