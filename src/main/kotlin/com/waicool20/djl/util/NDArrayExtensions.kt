package com.waicool20.djl.util

import ai.djl.ndarray.NDArray

fun NDArray.fEllipsis() = ":,".repeat(shape.shape.size - 1)
fun NDArray.bEllipsis() = ",:".repeat(shape.shape.size - 1)

operator fun NDArray.plus(array: NDArray): NDArray = add(array)
operator fun NDArray.plus(number: Number): NDArray = add(number)
operator fun NDArray.minus(array: NDArray): NDArray = sub(array)
operator fun NDArray.minus(number: Number): NDArray = sub(number)
operator fun NDArray.times(array: NDArray): NDArray = mul(array)
operator fun NDArray.times(number: Number): NDArray = mul(number)

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